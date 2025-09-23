import asyncio
import json
import os
import glob
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


class MediaIngestionPipeline:
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
    
    async def generate_embedding(self, media_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate embedding for a single image or video"""
        try:
            # Extract prompt from metadata
            metadata = json.loads(media_data.get('metadata', '{}'))
            prompt = metadata.get('prompt', {})
            
            if isinstance(prompt, dict):
                prompt_text = prompt.get('original', '') or prompt.get('enhanced', '')
            else:
                prompt_text = str(prompt)
            
            # Get output URL
            output_url = media_data.get('output_url', '')
            if not output_url:
                return None
            
            # Determine if it's an image or video
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff']
            video_extensions = ['.mp4', '.webm', '.mov', '.avi', '.mkv']
            
            is_image = any(output_url.lower().endswith(ext) for ext in image_extensions)
            is_video = any(output_url.lower().endswith(ext) for ext in video_extensions)
            
            if not is_image and not is_video:
                print(f"Skipping unknown file type: {output_url}")
                return None
            
            # Prepare request for FAL app
            request_data = {
                "text": prompt_text,
                "max_pixels": 360 * 420,
                "fps": 1.0
            }
            
            if is_image:
                request_data["image_url"] = output_url
                media_type = "image"
            else:
                request_data["video_url"] = output_url
                media_type = "video"
            
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
                    print(f"✓ {media_type} {media_data.get('id')}: embedding dimension {len(embedding)}")
                    
                    return {
                        "id": str(media_data.get('idx', media_data.get('id', ''))),
                        "embedding": embedding,
                        "text": prompt_text,
                        "imageUrl": output_url if is_image else "",
                        "videoUrl": output_url if is_video else "",
                        "createdAt": datetime.fromisoformat(
                            media_data.get('created_at', '').replace('+00', '')
                        ).timestamp() if media_data.get('created_at') else datetime.now().timestamp(),
                    }
                else:
                    print(f"✗ Error for {media_type} {media_data.get('id')}: {response.status}")
                    error_text = await response.text()
                    print(f"  Error details: {error_text[:200]}")
                    return None
                    
        except Exception as e:
            print(f"✗ Exception for {media_type} {media_data.get('id')}: {str(e)}")
            return None
    
    async def process_batch(self, media_items: List[Dict[str, Any]], batch_size: int = 5) -> Dict[str, int]:
        """Process a batch of media items"""
        results = {"success": 0, "failed": 0, "skipped": 0}
        embeddings_to_insert = []
        
        # Process in smaller concurrent batches
        for i in range(0, len(media_items), batch_size):
            batch = media_items[i:i + batch_size]
            
            # Create tasks for concurrent processing
            tasks = [self.generate_embedding(item) for item in batch]
            
            # Wait for all tasks in batch to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for item, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    print(f"Error processing media {item.get('id')}: {result}")
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
    
    async def ingest_from_json_files(self, json_pattern: str, limit: Optional[int] = None, 
                                   batch_size: int = 5, filter_platform: Optional[str] = None):
        """Ingest media from multiple JSON files"""
        # Find all matching JSON files
        json_files = sorted(glob.glob(json_pattern))
        
        if not json_files:
            print(f"No files found matching pattern: {json_pattern}")
            return {"success": 0, "failed": 0, "skipped": 0}
        
        print(f"Found {len(json_files)} JSON files to process:")
        for file in json_files:
            print(f"  - {file}")
        
        all_media = []
        
        # Load all JSON files
        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    items = json.load(f)
                print(f"Loaded {len(items)} records from {json_path}")
                all_media.extend(items)
            except Exception as e:
                print(f"Error loading {json_path}: {e}")
        
        print(f"\nTotal records loaded: {len(all_media)}")
        
        # Filter media items
        filtered_media = []
        
        for item in all_media:
            # Skip if no output URL
            output_url = item.get('output_url', '')
            if not output_url:
                continue
                
            # Skip failed/error items
            if item.get('status') != 'completed' or item.get('error'):
                continue
                
            # Apply platform filter if specified
            if filter_platform and item.get('platform') != filter_platform:
                continue
                
            filtered_media.append(item)
        
        print(f"Filtered to {len(filtered_media)} completed items")
        
        # Apply limit if specified
        if limit:
            filtered_media = filtered_media[:limit]
            print(f"Limited to {limit} items")
        
        # Group by platform and media type for stats
        platform_counts = defaultdict(int)
        file_type_counts = defaultdict(int)
        media_type_counts = {"images": 0, "videos": 0, "unknown": 0}
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff']
        video_extensions = ['.mp4', '.webm', '.mov', '.avi', '.mkv']
        
        for item in filtered_media:
            platform_counts[item.get('platform', 'unknown')] += 1
            
            # Count file types and media types
            url = item.get('output_url', '')
            extension = os.path.splitext(url.lower())[1]
            file_type_counts[extension] += 1
            
            if any(url.lower().endswith(ext) for ext in image_extensions):
                media_type_counts["images"] += 1
            elif any(url.lower().endswith(ext) for ext in video_extensions):
                media_type_counts["videos"] += 1
            else:
                media_type_counts["unknown"] += 1
        
        print("\nItems by platform:")
        for platform, count in platform_counts.items():
            print(f"  {platform}: {count}")
            
        print("\nItems by media type:")
        for media_type, count in media_type_counts.items():
            print(f"  {media_type}: {count}")
            
        print("\nItems by file extension:")
        for file_type, count in file_type_counts.items():
            print(f"  {file_type}: {count}")
        
        # Process media items
        print(f"\nStarting processing with batch size {batch_size}...")
        results = await self.process_batch(filtered_media, batch_size)
        
        return results


async def main():
    parser = argparse.ArgumentParser(description='Ingest images and videos to generate embeddings')
    parser.add_argument('json_pattern', help='Pattern for JSON files (e.g., "generation_rows*.json")')
    
    parser.add_argument('--fal-endpoint', default=os.getenv('FAL_ENDPOINT'), help='FAL endpoint URL')
    parser.add_argument('--fal-key', default=os.getenv('FAL_KEY'), help='FAL API key')
    parser.add_argument('--lancedb-uri', default=os.getenv('LANCEDB_URI'), help='LanceDB URI')
    parser.add_argument('--lancedb-key', default=os.getenv('LANCEDB_API_KEY'), help='LanceDB API key')
    
    parser.add_argument('--limit', type=int, help='Limit number of items to process')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for concurrent processing')
    parser.add_argument('--platform', choices=['fal', 'vertex_ai'], help='Filter by platform')
    
    args = parser.parse_args()
    
    async with MediaIngestionPipeline(
        fal_endpoint=args.fal_endpoint,
        fal_key=args.fal_key,
        lancedb_uri=args.lancedb_uri,
        lancedb_api_key=args.lancedb_key
    ) as pipeline:
        start_time = datetime.now()
        
        results = await pipeline.ingest_from_json_files(
            json_pattern=args.json_pattern,
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
        print(f"Rate: {results['success'] / duration:.2f} items/second" if duration > 0 else "N/A")


if __name__ == "__main__":
    asyncio.run(main())
