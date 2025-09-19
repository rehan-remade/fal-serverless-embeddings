import asyncio
import json
import os
from typing import List, Dict, Any, Optional
import aiohttp
from datetime import datetime
import lancedb
from pyarrow import schema as pa_schema, field as pa_field, float32, string, float64, list_ as pa_list_
import numpy as np
from tqdm.asyncio import tqdm
import argparse
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()


class VideoIngestionPipeline:
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
            # Note: We'll determine the actual embedding dimension from the first response
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
    
    async def generate_embedding(self, video_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate embedding for a single video"""
        try:
            # Extract prompt from metadata
            metadata = json.loads(video_data.get('metadata', '{}'))
            prompt = metadata.get('prompt', {})
            
            if isinstance(prompt, dict):
                prompt_text = prompt.get('original', '') or prompt.get('enhanced', '')
            else:
                prompt_text = str(prompt)
            
            # Skip if no video URL
            output_url = video_data.get('output_url', '')
            if not output_url:
                return None
            
            # Check if it's a video file (mp4, webm, mov, etc.)
            valid_video_extensions = ['.mp4', '.webm', '.mov', '.avi', '.mkv']
            if not any(output_url.lower().endswith(ext) for ext in valid_video_extensions):
                print(f"Skipping non-video file: {output_url}")
                return None
                
            # Prepare request for FAL app
            request_data = {
                "video_url": output_url,
                "text": prompt_text,
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
                    
                    # Debug: Check embedding dimension
                    print(f"Embedding dimension for video {video_data.get('id')}: {len(embedding)}")
                    
                    # Verify correct dimension
                    if len(embedding) != 1536:
                        print(f"WARNING: Expected 1536 dimensions, got {len(embedding)}")
                        # You might want to pad or truncate here if needed
                    
                    return {
                        "id": str(video_data.get('idx', video_data.get('id', ''))),
                        "embedding": embedding,
                        "text": prompt_text,
                        "imageUrl": "",  # Empty since this is for videos
                        "videoUrl": output_url,
                        "createdAt": datetime.fromisoformat(
                            video_data.get('created_at', '').replace('+00', '')
                        ).timestamp() if video_data.get('created_at') else datetime.now().timestamp(),
                    }
                else:
                    print(f"Error for video {video_data.get('id')}: {response.status}")
                    error_text = await response.text()
                    print(f"Error details: {error_text[:200]}")
                    return None
                    
        except Exception as e:
            print(f"Exception for video {video_data.get('id')}: {str(e)}")
            return None
    
    async def process_batch(self, videos: List[Dict[str, Any]], batch_size: int = 5) -> Dict[str, int]:
        """Process a batch of videos"""
        results = {"success": 0, "failed": 0, "skipped": 0}
        embeddings_to_insert = []
        
        # Process in smaller concurrent batches
        for i in range(0, len(videos), batch_size):
            batch = videos[i:i + batch_size]
            
            # Create tasks for concurrent processing
            tasks = [self.generate_embedding(video) for video in batch]
            
            # Wait for all tasks in batch to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for video, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    print(f"Error processing video {video.get('id')}: {result}")
                    results["failed"] += 1
                elif result is None:
                    results["skipped"] += 1
                else:
                    embeddings_to_insert.append(result)
                    results["success"] += 1
            
            # Insert batch into database
            if embeddings_to_insert:
                try:
                    await self.table.add(embeddings_to_insert)
                    print(f"Inserted {len(embeddings_to_insert)} embeddings into database")
                    embeddings_to_insert = []
                except Exception as e:
                    print(f"Error inserting into database: {e}")
                    results["failed"] += len(embeddings_to_insert)
                    results["success"] -= len(embeddings_to_insert)
                    embeddings_to_insert = []
            
            # Small delay between batches
            await asyncio.sleep(1)
        
        return results
    
    async def ingest_from_json(self, json_path: str, limit: Optional[int] = None, 
                              batch_size: int = 5, filter_platform: Optional[str] = None):
        """Ingest videos from JSON file"""
        # Load JSON data with UTF-8 encoding
        with open(json_path, 'r', encoding='utf-8') as f:
            videos = json.load(f)
        
        print(f"Loaded {len(videos)} records from {json_path}")
        
        # Filter videos
        filtered_videos = []
        skipped_images = 0
        
        for video in videos:
            # Skip if no output URL
            output_url = video.get('output_url', '')
            if not output_url:
                continue
                
            # Skip image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff']
            if any(output_url.lower().endswith(ext) for ext in image_extensions):
                skipped_images += 1
                continue
            
            # Skip non-video files that aren't images either
            video_extensions = ['.mp4', '.webm', '.mov', '.avi', '.mkv']
            if not any(output_url.lower().endswith(ext) for ext in video_extensions):
                print(f"Skipping unknown file type: {output_url}")
                continue
                
            # Skip failed/error videos
            if video.get('status') != 'completed' or video.get('error'):
                continue
                
            # Apply platform filter if specified
            if filter_platform and video.get('platform') != filter_platform:
                continue
                
            filtered_videos.append(video)
        
        print(f"Filtered to {len(filtered_videos)} completed videos")
        print(f"Skipped {skipped_images} image files")
        
        # Apply limit if specified
        if limit:
            filtered_videos = filtered_videos[:limit]
            print(f"Limited to {limit} videos")
        
        # Group by platform for stats
        platform_counts = defaultdict(int)
        file_type_counts = defaultdict(int)
        
        for video in filtered_videos:
            platform_counts[video.get('platform', 'unknown')] += 1
            # Count file types
            url = video.get('output_url', '')
            extension = os.path.splitext(url.lower())[1]
            file_type_counts[extension] += 1
        
        print("\nVideos by platform:")
        for platform, count in platform_counts.items():
            print(f"  {platform}: {count}")
            
        print("\nVideos by file type:")
        for file_type, count in file_type_counts.items():
            print(f"  {file_type}: {count}")
        
        # Process videos
        print(f"\nStarting processing with batch size {batch_size}...")
        results = await self.process_batch(filtered_videos, batch_size)
        
        return results

async def main():
    parser = argparse.ArgumentParser(description='Ingest videos and generate embeddings')
    parser.add_argument('json_path', help='Path to JSON file with video data')

    parser.add_argument('--fal-endpoint', default=os.getenv('FAL_ENDPOINT'), help='FAL endpoint URL')
    parser.add_argument('--fal-key', default=os.getenv('FAL_KEY'), help='FAL API key')
    parser.add_argument('--lancedb-uri', default=os.getenv('LANCEDB_URI'), help='LanceDB URI')
    parser.add_argument('--lancedb-key', default=os.getenv('LANCEDB_API_KEY'), help='LanceDB API key')

    parser.add_argument('--limit', type=int, help='Limit number of videos to process')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for concurrent processing')
    parser.add_argument('--platform', choices=['fal', 'vertex_ai'], help='Filter by platform')
    
    args = parser.parse_args()
    
    async with VideoIngestionPipeline(
        fal_endpoint=args.fal_endpoint,
        fal_key=args.fal_key,
        lancedb_uri=args.lancedb_uri,
        lancedb_api_key=args.lancedb_key
    ) as pipeline:
        start_time = datetime.now()
        
        results = await pipeline.ingest_from_json(
            json_path=args.json_path,
            limit=args.limit,
            batch_size=args.batch_size,
            filter_platform=args.platform
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n{'='*50}")
        print("INGESTION COMPLETE")
        print(f"{'='*50}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Success: {results['success']}")
        print(f"Failed: {results['failed']}")
        print(f"Skipped: {results['skipped']}")
        print(f"Rate: {results['success'] / duration:.2f} videos/second" if duration > 0 else "N/A")

if __name__ == "__main__":
    asyncio.run(main())
