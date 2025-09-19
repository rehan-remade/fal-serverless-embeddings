import asyncio
import os
import sys
import tempfile
import traceback
import uuid
from typing import Any, Optional, List
import json  # Add this import

import fal
from fal.toolkit import FAL_PERSISTENT_DIR
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from fastapi import HTTPException, Request, Response
from pydantic import BaseModel

from fal_app.constants import (
    EXAMPLE_TEXT,
    MAX_IMAGE_FILE_SIZE,
    MAX_VIDEO_FILE_SIZE,
)
from fal_app.models import (
    VLM2VecEmbeddingRequest,
    VLM2VecEmbeddingResponse,
)

import io


def get_flash_attn_wheel():
    """Get the appropriate flash-attn wheel for the current Python version"""
    import sys
    
    python_minor = sys.version_info.minor
    wheels = {
        10: "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
        11: "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
        12: "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    }
    
    if python_minor not in wheels:
        raise ValueError(f"No flash-attn wheel available for Python 3.{python_minor}")
    
    return wheels[python_minor]


def safe_snapshot_download(
    repo_id: str,
    revision: str = None,
    **kwargs: Any,
) -> str:
    """
    Safe snapshot download with local cache check first
    """
    try:
        print(f"Loading local repo: {repo_id}...")
        repo_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_files_only=True,
            **kwargs,
        )
        print(f"Loaded from local cache: {repo_path}")
    except LocalEntryNotFoundError:
        print(f"Local cache not found, downloading {repo_id} from Hugging Face Hub...")
        
        # Set high-performance download options
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "3600"
        
        # Additional performance settings
        if "max_workers" not in kwargs:
            kwargs["max_workers"] = 32
            
        repo_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_files_only=False,
            **kwargs,
        )
        print(f"Downloaded to: {repo_path}")
    return repo_path


def clone_vlm2vec_repo(commit_hash: str = "main") -> str:
    """Clone VLM2Vec repository with proper path handling"""
    import subprocess
    import shutil
    
    repo_dir = os.path.join("/data", ".fal", "repos", "VLM2Vec")
    
    # Check if repo already exists
    if os.path.exists(repo_dir) and os.path.exists(os.path.join(repo_dir, ".git")):
        print(f"VLM2Vec repository already exists at {repo_dir}")
        # Check out the specific commit
        try:
            subprocess.run(["git", "checkout", commit_hash], cwd=repo_dir, check=True, capture_output=True)
            print(f"Checked out commit {commit_hash}")
        except subprocess.CalledProcessError:
            print(f"Failed to checkout commit {commit_hash}, using existing checkout")
        return repo_dir
    
    # Clone the repository
    print("Cloning VLM2Vec repository...")
    temp_dir = f"/tmp/VLM2Vec_{uuid.uuid4().hex[:8]}"
    
    try:
        # Clone to temp directory first
        subprocess.run([
            "git", "clone", 
            "https://github.com/TIGER-AI-Lab/VLM2Vec.git",
            temp_dir
        ], check=True, capture_output=True)
        
        # Checkout specific commit
        subprocess.run([
            "git", "checkout", commit_hash
        ], cwd=temp_dir, check=True, capture_output=True)
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(repo_dir), exist_ok=True)
        
        # Move to final location
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        shutil.move(temp_dir, repo_dir)
        
        print(f"VLM2Vec repository cloned to {repo_dir}")
        return repo_dir
        
    except Exception as e:
        print(f"Error cloning repository: {e}")
        # Clean up temp directory if it exists
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise


class VLM2Vec(fal.App):
    """VLM2Vec-V2.0 fal serverless app for multimodal embedding generation"""
    
    machine_type = ["GPU-A100"]
    num_gpus = 1
    local_python_modules = ["fal_app"]
    requirements = [
        "torch==2.5.1",
        "torchvision",
        "numpy==1.26.4",
        "transformers==4.52.3",
        "huggingface-hub>=0.20.0",
        "hf-transfer",
        "peft",
        "accelerate",
        "datasets",
        "scipy",
        "tqdm",
        "wandb",
        "pillow",
        "wrapt",
        "py-cpuinfo",    
        "hjson",
        "scikit-learn",
        "scikit-image",
        "qwen_vl_utils",
        "ray",   
        "timm",
        "sentencepiece",
        "opencv-python",
        "decord",
        "hnswlib",
        "einops",
        "ninja",
        "opencv-contrib-python",
        f"flash-attn @ {get_flash_attn_wheel()}",
        "qwen-vl-utils[decord]==0.0.8",
        "aiohttp",
    ]

    # Model configuration
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    checkpoint_id: str = "TIGER-Lab/VLM2Vec-Qwen2VL-2B"
    vlm2vec_repo_id: str = "TIGER-AI-Lab/VLM2Vec"
    vlm2vec_commit_hash: str = "6ac33bb7e3049a0054a5952090f02ee5348fc4ed"

    async def setup(self) -> None:
        """Setup the VLM2Vec model and processor"""
        import torch
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Set Hugging Face token if available
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        
        # Download model weights with optimized settings
        self.model_dir = safe_snapshot_download(
            self.model_id,
            local_dir=os.path.join("/data", "models", "vlm2vec", self.model_id.replace("/", "_")),
            token=hf_token,
        )
        
        self.checkpoint_dir = safe_snapshot_download(
            self.checkpoint_id,
            local_dir=os.path.join("/data", "models", "vlm2vec", self.checkpoint_id.replace("/", "_")),
            token=hf_token,
        )
        
        # Clone VLM2Vec repository
        self.repo_dir = clone_vlm2vec_repo(self.vlm2vec_commit_hash)
        sys.path.insert(0, self.repo_dir)
        
        try:
            print("Importing VLM2Vec modules...")
            from src.arguments import ModelArguments, DataArguments
            from src.model.model import MMEBModel
            from src.model.processor import load_processor, QWEN2_VL, VLM_VIDEO_TOKENS
            from src.model.vlm_backbone.qwen2_vl.qwen_vl_utils import process_vision_info
            print("Imports successful")
            
            # Store imports
            self.QWEN2_VL = QWEN2_VL
            self.VLM_VIDEO_TOKENS = VLM_VIDEO_TOKENS
            self.process_vision_info = process_vision_info
            
            # Initialize model arguments
            print("Creating model arguments...")
            self.model_args = ModelArguments(
                model_name=self.model_dir,
                checkpoint_path=self.checkpoint_dir,
                pooling='last',
                normalize=True,
                model_backbone='qwen2_vl',
                lora=True
            )
            self.data_args = DataArguments()
            
            # Load processor and model
            print("Loading processor...")
            self.processor = load_processor(self.model_args, self.data_args)
            
            print("Loading model...")
            self.model = MMEBModel.load(self.model_args)
            
            print("Moving model to GPU...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device, dtype=torch.bfloat16)
            self.model.eval()
            
            print("Model setup complete")
            
            # Warmup with text only
            print("Running warmup...")
            warmup_request = VLM2VecEmbeddingRequest(
                text="This is a warmup test for the VLM2Vec model.",
            )
            embedding = await self.generate_embedding(warmup_request)
            print(f"Warmup complete! Generated embedding with {len(embedding)} dimensions")
            
        except Exception as e:
            print(f"Setup failed: {str(e)}")
            print(f"Traceback:\n{traceback.format_exc()}")
            raise

    async def generate_embedding(
        self,
        input: VLM2VecEmbeddingRequest,
    ) -> List[float]:
        """Generate embedding for given input"""
        import torch
        from PIL import Image
        import aiohttp
        
        # Add detailed logging
        print("=" * 50)
        print("INCOMING REQUEST:")
        print(f"Text: {input.text}")
        print(f"Image URL: {input.image_url}")
        print(f"Video URL: {input.video_url}")
        print(f"Max pixels: {input.max_pixels}")
        print(f"FPS: {input.fps}")
        print("=" * 50)
        
        try:
            with torch.no_grad():
                if input.video_url:
                    print("Processing video...")
                    # Download and process video
                    async with aiohttp.ClientSession() as session:
                        async with session.get(input.video_url) as response:
                            print(f"Video download status: {response.status}")
                            print(f"Video content type: {response.headers.get('content-type')}")
                            video_data = await response.read()
                            print(f"Video size: {len(video_data)} bytes")
                            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                                tmp.write(video_data)
                                video_path = tmp.name
                    
                    messages = [{
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_path,
                                "max_pixels": input.max_pixels,
                                "fps": input.fps,
                            },
                            {"type": "text", "text": input.text or "Represent the given video."},
                        ],
                    }]
                    
                    print(f"Messages structure: {json.dumps(messages, indent=2)}")
                    
                    image_inputs, video_inputs = self.process_vision_info(messages)
                    print(f"Image inputs: {image_inputs}")
                    print(f"Video inputs type: {type(video_inputs)}")
                    
                    inputs = self.processor(
                        text=f'{self.VLM_VIDEO_TOKENS[self.QWEN2_VL]} Represent the given video.',
                        videos=video_inputs,
                        return_tensors="pt"
                    )
                    
                    print("Processor output keys:", inputs.keys())
                    for key, value in inputs.items():
                        if hasattr(value, 'shape'):
                            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                        else:
                            print(f"  {key}: {type(value)}")
                    
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    inputs['pixel_values_videos'] = inputs['pixel_values_videos'].unsqueeze(0)
                    inputs['video_grid_thw'] = inputs['video_grid_thw'].unsqueeze(0)
                    
                    print("After unsqueeze:")
                    for key, value in inputs.items():
                        if hasattr(value, 'shape'):
                            print(f"  {key}: shape {value.shape}")
                    
                    output = self.model(qry=inputs)["qry_reps"]
                    
                    # Clean up
                    os.unlink(video_path)
                    
                elif input.image_url:
                    print("Processing image...")
                    # Download and process image
                    async with aiohttp.ClientSession() as session:
                        async with session.get(input.image_url) as response:
                            print(f"Image download status: {response.status}")
                            print(f"Image content type: {response.headers.get('content-type')}")
                            image_data = await response.read()
                            print(f"Image size: {len(image_data)} bytes")
                            image = Image.open(io.BytesIO(image_data)).convert('RGB')
                            print(f"Image dimensions: {image.size}")
                            print(f"Image mode: {image.mode}")
                    
                    # Try using the same approach as video processing
                    messages = [{
                        "role": "user", 
                        "content": [
                            {
                                "type": "image",
                                "image": image,  # Pass PIL Image directly
                                "max_pixels": input.max_pixels,
                            },
                            {"type": "text", "text": input.text or "Represent the given image."},
                        ],
                    }]
                    
                    print(f"Messages structure: {json.dumps([{**m, 'content': [{'type': c['type']} if 'type' in c else c for c in m['content']]} for m in messages], indent=2)}")
                    
                    try:
                        image_inputs, video_inputs = self.process_vision_info(messages)
                        print(f"Image inputs type: {type(image_inputs)}, length: {len(image_inputs) if hasattr(image_inputs, '__len__') else 'N/A'}")
                        print(f"Video inputs: {video_inputs}")
                        
                        # Process with images parameter
                        inputs = self.processor(
                            text=input.text or "Represent the given image.",
                            images=image_inputs if image_inputs else [image],
                            return_tensors="pt"
                        )
                    except Exception as e:
                        print(f"Error in process_vision_info: {e}")
                        print("Falling back to direct image processing")
                        inputs = self.processor(
                            text=input.text or "Represent the given image.",
                            images=[image],
                            return_tensors="pt"
                        )
                    
                    print("Processor output keys:", inputs.keys())
                    for key, value in inputs.items():
                        if hasattr(value, 'shape'):
                            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                        else:
                            print(f"  {key}: {type(value)}")
                    
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    
                    print("About to call model with inputs:")
                    for key, value in inputs.items():
                        if hasattr(value, 'shape'):
                            print(f"  {key}: shape {value.shape} on device {value.device}")
                    
                    output = self.model(qry=inputs)["qry_reps"]
                    
                elif input.text:
                    print("Processing text only...")
                    # Process text only
                    inputs = self.processor(
                        text=input.text,
                        images=None,
                        return_tensors="pt"
                    )
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    output = self.model(tgt=inputs)["tgt_reps"]
                    
                else:
                    raise ValueError("At least one input (text, image_url, or video_url) must be provided")
                
                # Convert to list
                # Fix: Convert bfloat16 to float32 before numpy conversion
                embedding = output.cpu().float().numpy().flatten().tolist()
                return embedding
                
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            print(f"Traceback:\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

    @fal.endpoint("/embed")
    async def embed(
        self, 
        input: VLM2VecEmbeddingRequest, 
        request: Request, 
        response: Response
    ) -> VLM2VecEmbeddingResponse:
        """Generate embedding for given input"""
        
        embedding = await self.generate_embedding(input)
        
        return VLM2VecEmbeddingResponse(
            embedding=embedding,
            dimension=len(embedding)
        )

    @fal.endpoint("/")
    async def root(
        self, 
        input: VLM2VecEmbeddingRequest, 
        request: Request, 
        response: Response
    ) -> VLM2VecEmbeddingResponse:
        """Default endpoint for embedding generation"""
        return await self.embed(input, request, response)


if __name__ == "__main__":
    app = fal.wrap_app(VLM2Vec)
    app()