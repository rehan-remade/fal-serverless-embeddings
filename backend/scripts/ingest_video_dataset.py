import asyncio
import os
import tempfile
import shutil
from typing import List, Dict, Any, Optional
import aiohttp
from datetime import datetime
import lancedb
from pyarrow import schema as pa_schema, field as pa_field, float32, string, float64, list_ as pa_list_
import numpy as np
from tqdm.asyncio import tqdm
import argparse
from dotenv import load_dotenv
from datasets import load_dataset
import json

load_dotenv()


class VideoDatasetIngestionPipeline:
    def __init__(self, fal_endpoint: str, fal_key: str, lancedb_uri: str, lancedb_api_key: str):
        self.fal_endpoint = fal_endpoint
        self.fal_key = fal_key
        self.lancedb_uri = lancedb_uri
        self.lancedb_api_key = lancedb_api_key
        self.db = None
        self.table = None
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        await self.connect_db()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def connect_db(self):
        """Connect to LanceDB and create/open table"""
        self.db = await lancedb.connect_async(
            uri=self.lancedb_uri,
            api_key=self.lancedb_api_key,
            region="us-east-1"
        )
        
        # Check if table exists
        tables = await self.db.table_names()
        
        if "embeddings" not in tables:
            # Create table with the correct schema
            schema = pa_schema([
                pa_field("id", string()),
                pa_field("embedding", pa_list_(float32(), 1536)),  # VLM2Vec-Qwen2VL-2B outputs 1536
                pa_field("text", string()),
                pa_field("imageUrl", string()),
                pa_field("videoUrl", string()),
                pa_field("createdAt", float64()),
            ])
            
            self.table = await self.db.create_table(
                "embeddings",
                schema=schema
            )
            print("Created new table: embeddings")
        else:
            self.table = await self.db.open_table("embeddings")
            print("Opened existing table: embeddings")
    
    async def upload_to_file_io(self, video_path: str) -> Optional[str]:
        """Upload video to file.io (temporary file hosting) and return URL"""
        try:
            with open(video_path, 'rb') as f:
                files = {'file': f}
                async with self.session.post('https://file.io', data=files) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get('success'):
                            return result['link']
            return None
        except Exception as e:
            print(f"Error uploading to file.io: {e}")
            return None
    
    async def generate_embedding(self, video_url: str, caption: str) -> Optional[Dict[str, Any]]:
        """Generate embedding for a video URL with caption using your FAL endpoint"""
        try:
            # Prepare request for FAL app
            request_data = {
                "video_url": video_url,
                "text": caption,
                "max_pixels": 360 * 420,
                "fps": 1.0
            }
            
            # Call FAL app to generate embedding
            async with self.session.post(
                f"{self.fal_endpoint}/embed",
                json=request_data,
                headers={
                    "Authorization": f"Key {self.fal_key}",
                    "Content-Type": "application/json"
                },
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    embedding = result['embedding']
                    
                    return {
                        "embedding": embedding,
                        "text": caption,
                        "videoUrl": video_url,
                    }
                else:
                    print(f"✗ Error generating embedding: {response.status}")
                    error_text = await response.text()
                    print(f"  Error details: {error_text[:200]}")
                    return None
                    
        except Exception as e:
            print(f"✗ Exception generating embedding: {str(e)}")
            return None
    
    async def process_video_entry(self, video_path: str, video_id: str, captions: List[str], dataset_name: str) -> Dict[str, int]:
        """Process a single video with its captions"""
        results = {"success": 0, "failed": 0}
        
        try:
            # Upload video to temporary storage
            print(f"Uploading video {video_id} to temporary storage...")
            video_url = await self.upload_to_file_io(video_path)
            
            if not video_url:
                print(f"✗ Failed to upload video {video_id}, skipping...")
                results["failed"] += len(captions[:3])
                return results
            
            print(f"✓ Uploaded video to: {video_url}")
            
            # Generate embeddings for captions
            embeddings_to_insert = []
            captions_to_process = captions[:3]  # Process fewer captions for testing
            
            for i, caption in enumerate(captions_to_process):
                try:
                    print(f"  Processing caption {i+1}/{len(captions_to_process)}: {caption[:50]}...")
                    
                    embedding_data = await self.generate_embedding(video_url, caption)
                    
                    if embedding_data:
                        record = {
                            "id": f"{dataset_name}_{video_id}_caption_{i}",
                            "embedding": embedding_data["embedding"],
                            "text": caption,
                            "imageUrl": "",
                            "videoUrl": video_url,
                            "createdAt": datetime.now().timestamp(),
                        }
                        embeddings_to_insert.append(record)
                        results["success"] += 1
                    else:
                        results["failed"] += 1
                        
                except Exception as e:
                    print(f"    ✗ Exception: {str(e)}")
                    results["failed"] += 1
                
                await asyncio.sleep(0.5)
            
            # Insert embeddings
            if embeddings_to_insert:
                try:
                    await self.table.add(embeddings_to_insert)
                    print(f"✓ Inserted {len(embeddings_to_insert)} embeddings")
                except Exception as e:
                    print(f"✗ Error inserting: {e}")
                    results["failed"] += len(embeddings_to_insert)
                    results["success"] -= len(embeddings_to_insert)
                    
        except Exception as e:
            print(f"✗ Error processing video {video_id}: {str(e)}")
            results["failed"] += len(captions[:3])
            
        return results
    
    def find_video_file(self, video_path: str) -> Optional[str]:
        """Find video file in various possible locations"""
        # If it's already a valid path, return it
        if os.path.exists(video_path):
            return video_path
        
        # Common HuggingFace cache locations
        hf_cache_dirs = [
            os.path.expanduser("~/.cache/huggingface/datasets"),
            os.path.expanduser("~/.cache/huggingface/hub"),
            os.path.expanduser("~/.cache/huggingface/datasets/downloads"),
            os.path.expanduser("~/.cache/huggingface/datasets/extracted"),
        ]
        
        # Try to find the file in cache directories
        for cache_dir in hf_cache_dirs:
            if os.path.exists(cache_dir):
                # Search recursively for the video file
                for root, dirs, files in os.walk(cache_dir):
                    if os.path.basename(video_path) in files:
                        full_path = os.path.join(root, os.path.basename(video_path))
                        if os.path.exists(full_path):
                            return full_path
        
        return None
    
    async def ingest_dataset(self, dataset_name: str, split: str = "train", limit: Optional[int] = None):
        """Ingest video dataset from HuggingFace"""
        print(f"Loading {dataset_name} dataset (split: {split})...")
        
        # Load dataset
        try:
            ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
            
            # Print first example to understand structure
            if len(ds) > 0:
                first_example = ds[0]
                print("\nFirst example structure:")
                for key, value in first_example.items():
                    print(f"  {key}: {type(value)}")
                    if isinstance(value, dict):
                        print(f"    Keys: {list(value.keys())}")
                    elif isinstance(value, str):
                        print(f"    Value (truncated): {value[:100]}...")
                        
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Make sure you're authenticated with HuggingFace using 'huggingface-cli login'")
            raise
        
        print(f"\nDataset loaded: {len(ds)} entries")
        
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
            print(f"Limited to {len(ds)} entries")
        
        total_results = {"success": 0, "failed": 0}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for idx, example in enumerate(tqdm(ds, desc="Processing videos")):
                # Handle different dataset structures
                if dataset_name == "friedrichor/MSR-VTT":
                    video_id = example.get('video_id', f"video_{idx}")
                    video_data = example.get('video', None)
                    captions = [example.get('caption', "")]  # MSR-VTT might have single caption
                elif dataset_name == "VLM2Vec/MSVD":
                    video_id = example['video_id']
                    video_data = example['video']
                    captions = example['caption']
                else:
                    # Generic handling
                    video_id = example.get('video_id', example.get('id', f"video_{idx}"))
                    video_data = example.get('video', example.get('video_path', None))
                    captions = example.get('caption', example.get('captions', []))
                    if isinstance(captions, str):
                        captions = [captions]
                
                print(f"\nProcessing video {idx+1}/{len(ds)}: {video_id}")
                print(f"  Captions: {len(captions)}")
                print(f"  Video data type: {type(video_data)}")
                
                # Save video to temporary file
                video_path = os.path.join(temp_dir, f"{video_id}.mp4")
                
                try:
                    # Handle different types of video data
                    if isinstance(video_data, str):
                        # Video data is a file path
                        found_path = self.find_video_file(video_data)
                        if found_path:
                            shutil.copy(found_path, video_path)
                        else:
                            print(f"  Video file not found: {video_data}")
                            continue
                            
                    elif isinstance(video_data, bytes):
                        with open(video_path, 'wb') as f:
                            f.write(video_data)
                            
                    elif hasattr(video_data, 'read'):
                        # If it's a file-like object
                        with open(video_path, 'wb') as f:
                            f.write(video_data.read())
                            
                    elif isinstance(video_data, dict):
                        # Handle dict structures
                        if 'bytes' in video_data:
                            with open(video_path, 'wb') as f:
                                f.write(video_data['bytes'])
                        elif 'path' in video_data:
                            found_path = self.find_video_file(video_data['path'])
                            if found_path:
                                shutil.copy(found_path, video_path)
                            else:
                                print(f"  Video file not found: {video_data['path']}")
                                continue
                        else:
                            print(f"  Unknown video data dict structure: {list(video_data.keys())}")
                            continue
                    else:
                        print(f"  Unsupported video data type: {type(video_data)}")
                        continue
                    
                    # Verify file was created and has content
                    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                        print(f"  Video file is empty or wasn't created")
                        continue
                    
                    # Process video
                    dataset_short_name = dataset_name.split('/')[-1].lower()
                    results = await self.process_video_entry(video_path, video_id, captions, dataset_short_name)
                    total_results["success"] += results["success"]
                    total_results["failed"] += results["failed"]
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()
                    total_results["failed"] += 1
                
                await asyncio.sleep(1)
        
        return total_results


async def main():
    parser = argparse.ArgumentParser(description='Ingest video dataset to generate embeddings')
    
    parser.add_argument('--dataset', default='VLM2Vec/MSVD', 
                       help='Dataset name (e.g., VLM2Vec/MSVD, friedrichor/MSR-VTT)')
    parser.add_argument('--fal-endpoint', default=os.getenv('FAL_ENDPOINT'), help='FAL endpoint URL')
    parser.add_argument('--fal-key', default=os.getenv('FAL_KEY'), help='FAL API key')
    parser.add_argument('--lancedb-uri', default=os.getenv('LANCEDB_URI'), help='LanceDB URI')
    parser.add_argument('--lancedb-key', default=os.getenv('LANCEDB_API_KEY'), help='LanceDB API key')
    
    parser.add_argument('--split', default='test', choices=['train', 'validation', 'test'], 
                       help='Dataset split to process')
    parser.add_argument('--limit', type=int, default=5, help='Limit number of videos')
    
    args = parser.parse_args()
    
    if not all([args.fal_endpoint, args.fal_key, args.lancedb_uri, args.lancedb_key]):
        print("Error: Missing required parameters. Please set environment variables:")
        print("  FAL_ENDPOINT, FAL_KEY, LANCEDB_URI, LANCEDB_API_KEY")
        return
    
    async with VideoDatasetIngestionPipeline(
        fal_endpoint=args.fal_endpoint,
        fal_key=args.fal_key,
        lancedb_uri=args.lancedb_uri,
        lancedb_api_key=args.lancedb_key
    ) as pipeline:
        start_time = datetime.now()
        
        results = await pipeline.ingest_dataset(
            dataset_name=args.dataset,
            split=args.split,
            limit=args.limit
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n{'='*50}")
        print("VIDEO DATASET INGESTION COMPLETE")
        print(f"{'='*50}")
        print(f"Dataset: {args.dataset}")
        print(f"Split: {args.split}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Success: {results['success']} embeddings")
        print(f"Failed: {results['failed']} embeddings")
        print(f"Rate: {results['success'] / duration:.2f} embeddings/second" if duration > 0 else "N/A")


if __name__ == "__main__":
    asyncio.run(main())
