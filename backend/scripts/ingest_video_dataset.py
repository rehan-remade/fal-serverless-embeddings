import asyncio
import os
from typing import Dict, Any, Optional, List
import aiohttp
from datetime import datetime
import lancedb
from pyarrow import schema as pa_schema, field as pa_field, float32, string, float64, list_ as pa_list_
from tqdm.asyncio import tqdm
import argparse
from dotenv import load_dotenv
from datasets import load_dataset
import time
import random

load_dotenv()


class PexelsVideoIngestionPipeline:
    def __init__(self, fal_endpoint: str, fal_key: str, lancedb_uri: str, lancedb_api_key: str):
        self.fal_endpoint = fal_endpoint
        self.fal_key = fal_key
        self.lancedb_uri = lancedb_uri
        self.lancedb_api_key = lancedb_api_key
        self.db = None
        self.table = None
        self.session = None
        
        # Rate limiting tracking
        self.rate_limit_reset_time = None
        self.consecutive_429s = 0
        
        # Cache for existing video IDs
        self.existing_video_ids = set()
        
        # Set FAL API key for fal_client
        os.environ["FAL_KEY"] = fal_key
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        await self.connect_db()
        await self.load_existing_video_ids()
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
    
    async def load_existing_video_ids(self):
        """Load all existing video IDs from the database"""
        print("Loading existing video IDs from database...")
        try:
            # Query all records that start with 'pexels_'
            results = await self.table.search() \
                .select(["id"]) \
                .where("id LIKE 'pexels_%'") \
                .to_list()
            
            # Extract video IDs
            for result in results:
                self.existing_video_ids.add(result['id'])
            
            print(f"Found {len(self.existing_video_ids)} existing Pexels videos in database")
        except Exception as e:
            print(f"Error loading existing IDs (might be empty table): {e}")
            # If table is empty or error, continue with empty set
            self.existing_video_ids = set()
    
    async def is_video_processed(self, video_id: str) -> bool:
        """Check if a video has already been processed"""
        pexels_id = f"pexels_{video_id}"
        return pexels_id in self.existing_video_ids

    async def check_video_url_accessibility(self, video_url: str) -> bool:
        """Check if video URL is accessible before sending to FAL"""
        try:
            async with self.session.head(video_url, allow_redirects=True, timeout=aiohttp.ClientTimeout(total=10)) as response:
                # Check if we're being rate limited
                if response.status == 429:
                    self.consecutive_429s += 1
                    print(f"\n⚠️ Rate limited by Pexels (429). Consecutive 429s: {self.consecutive_429s}")
                    return False
                
                # Reset counter on successful request
                self.consecutive_429s = 0
                
                # Check if content type is video
                content_type = response.headers.get('content-type', '')
                if response.status == 200 and ('video' in content_type or 'octet-stream' in content_type):
                    return True
                    
                print(f"  ⚠️ Invalid response: status={response.status}, content-type={content_type}")
                return False
                
        except Exception as e:
            print(f"  ⚠️ Error checking video URL: {str(e)}")
            return False

    async def generate_embedding_with_retry(self, video_url: str, caption: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Generate embedding with retry logic"""
        for attempt in range(max_retries):
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
                    elif response.status == 500:
                        # Internal server error might be due to invalid video data
                        error_text = await response.text()
                        if "429" in error_text or "Invalid data found" in error_text:
                            print(f"  ⚠️ FAL endpoint received invalid video data (likely rate limited HTML)")
                            return None
                        print(f"  ✗ FAL Error ({response.status}): {error_text[:200]}")
                        
                        # Retry with exponential backoff
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            print(f"  ⏳ Retrying in {wait_time} seconds...")
                            await asyncio.sleep(wait_time)
                            continue
                    else:
                        print(f"  ✗ Error generating embedding: {response.status}")
                        error_text = await response.text()
                        print(f"  Error details: {error_text[:200]}")
                    
                    return None
                        
            except Exception as e:
                print(f"  ✗ Exception generating embedding: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  ⏳ Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
                return None
        
        return None
    
    async def process_pexels_video_with_semaphore(
        self, 
        semaphore: asyncio.Semaphore,
        rate_limiter: asyncio.Semaphore,
        video_url: str, 
        video_id: str, 
        title: str, 
        thumbnail_url: str,
        idx: int,
        total: int
    ) -> Dict[str, int]:
        """Process a single Pexels video with concurrency and rate limiting control"""
        async with semaphore:
            # Check if video already processed
            if await self.is_video_processed(video_id):
                print(f"\n[{idx+1}/{total}] Skipping already processed video {video_id}")
                return {"success": 0, "failed": 0, "skipped": 1}
            
            # Additional rate limiting to prevent hitting Pexels limits
            async with rate_limiter:
                results = {"success": 0, "failed": 0, "skipped": 0}
                
                try:
                    print(f"\n[{idx+1}/{total}] Processing video {video_id}")
                    print(f"  Title: {title[:50]}...")
                    
                    embedding_data = await self.generate_embedding_with_retry(video_url, title)
                    
                    if embedding_data:
                        record = {
                            "id": f"pexels_{video_id}",
                            "embedding": embedding_data["embedding"],
                            "text": title,
                            "imageUrl": thumbnail_url,  # Use thumbnail URL
                            "videoUrl": video_url,
                            "createdAt": datetime.now().timestamp(),
                        }
                        
                        # Insert embedding
                        try:
                            await self.table.add([record])
                            print(f"  ✓ Success: {title[:50]}...")
                            results["success"] += 1
                            # Add to cache
                            self.existing_video_ids.add(f"pexels_{video_id}")
                        except Exception as e:
                            print(f"  ✗ Error inserting: {e}")
                            results["failed"] += 1
                    else:
                        print(f"  ✗ Failed to generate embedding")
                        results["failed"] += 1
                        
                except Exception as e:
                    print(f"  ✗ Error processing video: {str(e)}")
                    results["failed"] += 1
                    
                # Add delay between requests to avoid rate limiting
                await asyncio.sleep(0.5)  # 500ms between requests
                    
                return results

    async def ingest_pexels_dataset(
        self, 
        limit: Optional[int] = None, 
        duration_limit: int = 10, 
        concurrency: int = 5,
        requests_per_minute: int = 60,
        randomize: bool = True
    ):
        """Ingest Pexels-400k dataset with concurrent processing and rate limiting"""
        print(f"Loading Pexels-400k dataset...")
        
        # Load dataset
        try:
            # Pexels dataset doesn't have splits, so we use "train"
            ds = load_dataset("jovianzm/Pexels-400k", split="train", trust_remote_code=True)
            
            # Print first example to understand structure
            if len(ds) > 0:
                first_example = ds[0]
                print("\nFirst example structure:")
                for key, value in first_example.items():
                    print(f"  {key}: {type(value)}")
                    if isinstance(value, str):
                        print(f"    Value (truncated): {value[:100]}...")
                    else:
                        print(f"    Value: {value}")
                        
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Make sure you're authenticated with HuggingFace using 'huggingface-cli login'")
            raise
        
        print(f"\nDataset loaded: {len(ds)} entries")
        
        # Filter by duration (10 seconds or less)
        print(f"Filtering videos by duration (<= {duration_limit} seconds)...")
        filtered_ds = [ex for ex in ds if ex.get('duration', float('inf')) <= duration_limit]
        print(f"Filtered to {len(filtered_ds)} videos (<= {duration_limit}s)")
        
        # Randomize if requested
        if randomize:
            print("Randomizing dataset order...")
            random.shuffle(filtered_ds)
        
        # Apply limit if specified
        if limit and limit < len(filtered_ds):
            filtered_ds = filtered_ds[:limit]
            print(f"Limited to {len(filtered_ds)} entries")
        
        print(f"\nProcessing with concurrency: {concurrency}")
        print(f"Rate limit: {requests_per_minute} requests/minute")
        print(f"Randomization: {'ON' if randomize else 'OFF'}")
        
        # Create semaphores for concurrency and rate limiting
        concurrency_semaphore = asyncio.Semaphore(concurrency)
        
        # Create a rate limiter that allows requests_per_minute
        rate_limit_delay = 60.0 / requests_per_minute
        
        class RateLimiter:
            def __init__(self, delay):
                self.delay = delay
                self.last_request = 0
                
            async def __aenter__(self):
                now = time.time()
                time_since_last = now - self.last_request
                if time_since_last < self.delay:
                    await asyncio.sleep(self.delay - time_since_last)
                self.last_request = time.time()
                
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
        
        rate_limiter = RateLimiter(rate_limit_delay)
        
        # Prepare all tasks
        tasks = []
        skipped_no_url = 0
        skipped_existing = 0
        
        for idx, example in enumerate(filtered_ds):
            video_url = example.get('video', '')
            title = example.get('title', '')
            thumbnail_url = example.get('thumbnail', '')
            duration = example.get('duration', 0)
            
            # Extract video ID from URL
            video_id = video_url.split('/')[-1] if '/' in video_url else f"video_{idx}"
            
            # Skip if no video URL
            if not video_url:
                skipped_no_url += 1
                continue
            
            # Quick check if already processed (before creating task)
            if await self.is_video_processed(video_id):
                skipped_existing += 1
                continue
            
            # Create task for concurrent processing
            task = self.process_pexels_video_with_semaphore(
                concurrency_semaphore,
                rate_limiter,
                video_url,
                video_id,
                title,
                thumbnail_url,
                idx,
                len(filtered_ds)
            )
            tasks.append(task)
        
        print(f"\nPre-filtered results:")
        print(f"  - Skipped {skipped_existing} already processed videos")
        print(f"  - Skipped {skipped_no_url} videos without URLs")
        print(f"  - Will process {len(tasks)} new videos")
        
        if len(tasks) == 0:
            print("\nNo new videos to process!")
            return {"success": 0, "failed": 0, "skipped": skipped_existing + skipped_no_url}
        
        # Process all tasks concurrently with progress bar
        print(f"\nStarting concurrent processing of {len(tasks)} videos...")
        
        # Use tqdm to show overall progress
        results_list = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing videos"):
            result = await coro
            results_list.append(result)
        
        # Aggregate results
        total_results = {"success": 0, "failed": 0, "skipped": skipped_no_url + skipped_existing}
        for result in results_list:
            total_results["success"] += result["success"]
            total_results["failed"] += result["failed"]
            total_results["skipped"] += result.get("skipped", 0)
        
        return total_results


async def main():
    parser = argparse.ArgumentParser(description='Ingest Pexels-400k videos to generate embeddings')
    
    parser.add_argument('--fal-endpoint', default=os.getenv('FAL_ENDPOINT'), help='FAL endpoint URL')
    parser.add_argument('--fal-key', default=os.getenv('FAL_KEY'), help='FAL API key')
    parser.add_argument('--lancedb-uri', default=os.getenv('LANCEDB_URI'), help='LanceDB URI')
    parser.add_argument('--lancedb-key', default=os.getenv('LANCEDB_API_KEY'), help='LanceDB API key')
    parser.add_argument('--limit', type=int, default=5, help='Limit number of videos to process')
    parser.add_argument('--duration-limit', type=int, default=10, help='Maximum video duration in seconds')
    parser.add_argument('--concurrency', type=int, default=3, help='Number of concurrent requests (default: 3)')
    parser.add_argument('--rate-limit', type=int, default=30, help='Requests per minute (default: 30)')
    parser.add_argument('--no-randomize', action='store_true', help='Disable randomization of video order')
    
    args = parser.parse_args()
    
    if not all([args.fal_endpoint, args.fal_key, args.lancedb_uri, args.lancedb_key]):
        print("Error: Missing required parameters. Please set environment variables:")
        print("  FAL_ENDPOINT, FAL_KEY, LANCEDB_URI, LANCEDB_API_KEY")
        return
    
    async with PexelsVideoIngestionPipeline(
        fal_endpoint=args.fal_endpoint,
        fal_key=args.fal_key,
        lancedb_uri=args.lancedb_uri,
        lancedb_api_key=args.lancedb_key
    ) as pipeline:
        start_time = datetime.now()
        
        results = await pipeline.ingest_pexels_dataset(
            limit=args.limit,
            duration_limit=args.duration_limit,
            concurrency=args.concurrency,
            requests_per_minute=args.rate_limit,
            randomize=not args.no_randomize
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n{'='*50}")
        print("PEXELS VIDEO INGESTION COMPLETE")
        print(f"{'='*50}")
        print(f"Concurrency: {args.concurrency}")
        print(f"Rate limit: {args.rate_limit} requests/minute")
        print(f"Randomization: {'OFF' if args.no_randomize else 'ON'}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Success: {results['success']} embeddings")
        print(f"Failed: {results['failed']} embeddings")
        print(f"Skipped: {results['skipped']} videos")
        print(f"Rate: {results['success'] / duration:.2f} embeddings/second" if duration > 0 else "N/A")


if __name__ == "__main__":
    asyncio.run(main())
