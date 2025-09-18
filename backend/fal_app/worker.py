import os
import sys
import tempfile
import torch
from typing import Any, Optional, List
from PIL import Image

from fal_app.utils import DistributedWorker, get_seed
from fal_app.constants import DEFAULT_MAX_PIXELS, DEFAULT_FPS


class VLM2VecWorker(DistributedWorker):
    """Worker for VLM2Vec embedding generation"""
    
    def setup(
        self,
        model_path: str,
        checkpoint_path: str,
        repo_path: str,
        **kwargs: Any,
    ) -> None:
        """Setup the VLM2Vec model"""
        try:
            print(f"[VLM2VecWorker] Starting setup")
            print(f"[VLM2VecWorker] model_path: {model_path}")
            print(f"[VLM2VecWorker] checkpoint_path: {checkpoint_path}")
            print(f"[VLM2VecWorker] repo_path: {repo_path}")
            
            assert model_path is not None, "model_path must be provided"
            assert checkpoint_path is not None, "checkpoint_path must be provided"
            assert repo_path is not None, "repo_path must be provided"
            
            # Add repo to path
            sys.path.insert(0, repo_path)
            print(f"[VLM2VecWorker] Added {repo_path} to Python path")
            
            # Import VLM2Vec modules
            print("[VLM2VecWorker] Importing VLM2Vec modules...")
            from src.arguments import ModelArguments, DataArguments
            from src.model.model import MMEBModel
            from src.model.processor import load_processor, QWEN2_VL, VLM_VIDEO_TOKENS
            from src.model.vlm_backbone.qwen2_vl.qwen_vl_utils import process_vision_info
            print("[VLM2VecWorker] Imports successful")
            
            # Store imports
            self.QWEN2_VL = QWEN2_VL
            self.VLM_VIDEO_TOKENS = VLM_VIDEO_TOKENS
            self.process_vision_info = process_vision_info
            
            # Initialize model arguments
            print("[VLM2VecWorker] Creating model arguments")
            self.model_args = ModelArguments(
                model_name=model_path,
                checkpoint_path=checkpoint_path,
                pooling='last',
                normalize=True,
                model_backbone='qwen2_vl',
                lora=True
            )
            self.data_args = DataArguments()
            
            # Load processor and model
            print("[VLM2VecWorker] Loading processor...")
            self.processor = load_processor(self.model_args, self.data_args)
            
            print("[VLM2VecWorker] Loading model...")
            self.model = MMEBModel.load(self.model_args)
            
            print(f"[VLM2VecWorker] Moving model to device: {self.device}")
            self.model = self.model.to(self.device, dtype=torch.bfloat16)
            self.model.eval()
            
            print(f"[VLM2VecWorker] Setup complete for rank {self.rank}")
            
        except Exception as e:
            import traceback
            print(f"[VLM2VecWorker] Setup failed: {str(e)}")
            print(f"[VLM2VecWorker] Traceback:\n{traceback.format_exc()}")
            raise

    def __call__(
        self,
        text: Optional[str] = None,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        fps: float = DEFAULT_FPS,
        is_warmup: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate embedding for given input"""
        
        print(f"[VLM2VecWorker] Called with: text={text}, image_path={image_path}, video_path={video_path}")
        
        if self.rank != 0:
            print(f"[VLM2VecWorker] Rank {self.rank} != 0, returning empty")
            return {}
        
        try:
            with torch.no_grad():
                if video_path:
                    print("[VLM2VecWorker] Processing video input")
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video",
                                    "video": video_path,
                                    "max_pixels": max_pixels,
                                    "fps": fps,
                                },
                                {"type": "text", "text": text or "Represent the given video."},
                            ],
                        }
                    ]
                    
                    image_inputs, video_inputs = self.process_vision_info(messages)
                    inputs = self.processor(
                        text=f'{self.VLM_VIDEO_TOKENS[self.QWEN2_VL]} Represent the given video.',
                        videos=video_inputs,
                        return_tensors="pt"
                    )
                    print(f"[VLM2VecWorker] Processor returned keys for video: {list(inputs.keys())}")
                    
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    print("[VLM2VecWorker] Moved video inputs to device")
                    inputs['pixel_values_videos'] = inputs['pixel_values_videos'].unsqueeze(0)
                    inputs['video_grid_thw'] = inputs['video_grid_thw'].unsqueeze(0)
                    
                    output = self.model(qry=inputs)["qry_reps"]
                    print(f"[VLM2VecWorker] Model output shape for video: {output.shape}")
                    
                elif image_path:
                    print("[VLM2VecWorker] Processing image input")
                    # Handle image input
                    image = Image.open(image_path).convert('RGB')
                    inputs = self.processor(
                        text=text or "Represent the given image.",
                        images=[image],
                        return_tensors="pt"
                    )
                    print(f"[VLM2VecWorker] Processor returned keys for image: {list(inputs.keys())}")
                    
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    print("[VLM2VecWorker] Moved image inputs to device")
                    
                    output = self.model(qry=inputs)["qry_reps"]
                    print(f"[VLM2VecWorker] Model output shape for image: {output.shape}")
                    
                elif text:
                    print("[VLM2VecWorker] Processing text input")
                    # Handle text-only input
                    inputs = self.processor(
                        text=text,
                        images=None,
                        return_tensors="pt"
                    )
                    print(f"[VLM2VecWorker] Processor returned keys for text: {list(inputs.keys())}")
                    
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    print("[VLM2VecWorker] Moved text inputs to device")
                    
                    output = self.model(tgt=inputs)["tgt_reps"]
                    print(f"[VLM2VecWorker] Model output shape for text: {output.shape}")
                    
                else:
                    raise ValueError("At least one input (text, image_path, or video_path) must be provided")
                
                # Convert to list for JSON serialization
                embedding = output.cpu().numpy().flatten().tolist()
                print(f"[VLM2VecWorker] Generated embedding with {len(embedding)} dimensions")
                
                return {
                    "embedding": embedding
                }
                
        except Exception as e:
            import traceback
            error_msg = str(e)
            tb = traceback.format_exc()
            print(f"[VLM2VecWorker] Error during inference: {error_msg}")
            print(f"[VLM2VecWorker] Traceback:\n{tb}")
            return {
                "error": error_msg,
                "traceback": tb
            }


