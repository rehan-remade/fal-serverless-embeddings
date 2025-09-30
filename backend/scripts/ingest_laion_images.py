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
import hashlib
import random

load_dotenv()


class LAIONImageIngestionPipeline:
    def __init__(self, fal_endpoint: str, fal_key: str, lancedb_uri: str, lancedb_api_key: str):
        self.fal_endpoint = fal_endpoint
        self.fal_key = fal_key
        self.lancedb_uri = lancedb_uri
        self.lancedb_api_key = lancedb_api_key
        self.db = None
        self.table = None
        self.session = None
        
        # Cache for existing image IDs
        self.existing_image_ids = set()
        
        # Set FAL API key
        os.environ["FAL_KEY"] = fal_key
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        await self.connect_db()
        await self.load_existing_image_ids()
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
    
    async def load_existing_image_ids(self):
        """Load all existing LAION image IDs from the database"""
        print("Loading existing LAION image IDs from database...")
        try:
            # Query all records that start with 'laion_'
            results = await self.table.search() \
                .select(["id"]) \
                .where("id LIKE 'laion_%'") \
                .to_list()
            
            # Extract image IDs
            for result in results:
                self.existing_image_ids.add(result['id'])
            
            print(f"Found {len(self.existing_image_ids)} existing LAION images in database")
        except Exception as e:
            print(f"Error loading existing IDs (might be empty table): {e}")
            self.existing_image_ids = set()
    
    def generate_image_id(self, url: str) -> str:
        """Generate a unique ID for an image based on its URL"""
        return hashlib.md5(url.encode()).hexdigest()[:16]
    
    async def is_image_processed(self, image_id: str) -> bool:
        """Check if an image has already been processed"""
        laion_id = f"laion_{image_id}"
        return laion_id in self.existing_image_ids
    
    async def check_image_url_accessibility(self, image_url: str) -> bool:
        """Check if image URL is accessible"""
        try:
            async with self.session.head(
                image_url, 
                allow_redirects=True, 
                timeout=aiohttp.ClientTimeout(total=5),
                ssl=False  # Many LAION images have SSL issues
            ) as response:
                content_type = response.headers.get('content-type', '')
                if response.status == 200 and 'image' in content_type:
                    return True
                return False
        except Exception as e:
            # Many LAION URLs are dead, this is expected
            return False
    
    async def generate_embedding(self, image_url: str, caption: str) -> Optional[Dict[str, Any]]:
        """Generate embedding for an image URL with caption using FAL endpoint"""
        try:
            # Prepare request for FAL app
            request_data = {
                "image_url": image_url,
                "text": caption,
                "max_pixels": 360 * 420,
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
                        "imageUrl": image_url,
                    }
                else:
                    error_text = await response.text()
                    print(f"  ✗ Error generating embedding ({response.status}): {error_text[:200]}")
                    return None
                    
        except Exception as e:
            print(f"  ✗ Exception generating embedding: {str(e)}")
            return None
    
    async def process_laion_image_with_semaphore(
        self, 
        semaphore: asyncio.Semaphore,
        image_data: Dict[str, Any],
        idx: int,
        total: int
    ) -> Dict[str, int]:
        """Process a single LAION image with concurrency control"""
        async with semaphore:
            results = {"success": 0, "failed": 0, "skipped": 0}
            
            image_url = image_data.get('URL', '')
            caption = image_data.get('TEXT', '')
            aesthetic_score = image_data.get('AESTHETIC_SCORE', 0)
            width = image_data.get('WIDTH', 0)
            height = image_data.get('HEIGHT', 0)
            
            # Generate image ID
            image_id = self.generate_image_id(image_url)
            
            # Check if already processed
            if await self.is_image_processed(image_id):
                results["skipped"] += 1
                return results
            
            try:
                print(f"\n[{idx+1}/{total}] Processing image {image_id}")
                print(f"  Caption: {caption[:50]}...")
                print(f"  Aesthetic Score: {aesthetic_score:.2f}")
                print(f"  Dimensions: {width}x{height}")
                
                # Check if URL is accessible (optional, can skip for speed)
                # if not await self.check_image_url_accessibility(image_url):
                #     print(f"  ✗ Image URL not accessible")
                #     results["failed"] += 1
                #     return results
                
                embedding_data = await self.generate_embedding(image_url, caption)
                
                if embedding_data:
                    record = {
                        "id": f"laion_{image_id}",
                        "embedding": embedding_data["embedding"],
                        "text": caption,
                        "imageUrl": image_url,
                        "videoUrl": "",  # Empty for images
                        "createdAt": datetime.now().timestamp(),
                    }
                    
                    # Insert embedding
                    try:
                        await self.table.add([record])
                        print(f"  ✓ Success: {caption[:50]}...")
                        results["success"] += 1
                        # Add to cache
                        self.existing_image_ids.add(f"laion_{image_id}")
                    except Exception as e:
                        print(f"  ✗ Error inserting: {e}")
                        results["failed"] += 1
                else:
                    print(f"  ✗ Failed to generate embedding")
                    results["failed"] += 1
                    
            except Exception as e:
                print(f"  ✗ Error processing image: {str(e)}")
                results["failed"] += 1
                
            # Small delay between requests
            await asyncio.sleep(0.1)
                
            return results

    async def ingest_laion_dataset(
        self, 
        num_images: int = 1000,
        concurrency: int = 5,
        min_aesthetic_score: float = 5.0  # Lower threshold for random sampling
    ):
        """Ingest random LAION images"""
        print(f"Loading LAION aesthetics dataset...")
        
        # Load dataset
        try:
            ds = load_dataset("dclure/laion-aesthetics-12m-umap", split="train", trust_remote_code=True)
            
            # Print dataset info
            print(f"\nDataset loaded: {len(ds)} total entries")
            
            # Convert to pandas for easier filtering
            print("Filtering valid entries...")
            df = ds.to_pandas()
            
            # Filter out invalid entries
            df = df[df['AESTHETIC_SCORE'].notna()]
            df = df[df['URL'].notna()]
            df = df[df['TEXT'].notna()]
            df = df[df['AESTHETIC_SCORE'] >= min_aesthetic_score]
            
            print(f"Filtered to {len(df)} valid entries with aesthetic score >= {min_aesthetic_score}")
            
            # Random sampling
            print(f"Randomly sampling {num_images} images...")
            sample_size = min(num_images * 3, len(df))  # Sample 3x to account for dead URLs and duplicates
            df_sample = df.sample(n=sample_size, random_state=None)  # None for true randomness
            
            # Convert back to list of dicts
            images_to_process = df_sample.to_dict('records')
            
            print(f"Aesthetic score range in sample: {df_sample['AESTHETIC_SCORE'].min():.2f} - {df_sample['AESTHETIC_SCORE'].max():.2f}")
            print(f"Average aesthetic score: {df_sample['AESTHETIC_SCORE'].mean():.2f}")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
        
        print(f"\nProcessing {len(images_to_process)} randomly sampled images")
        print(f"Concurrency: {concurrency}")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrency)
        
        # Prepare tasks for unprocessed images
        tasks = []
        skipped_existing = 0
        
        for idx, image_data in enumerate(images_to_process):
            # Skip if already processed (quick check)
            image_id = self.generate_image_id(image_data['URL'])
            if await self.is_image_processed(image_id):
                skipped_existing += 1
                continue
                
            task = self.process_laion_image_with_semaphore(
                semaphore,
                image_data,
                idx,
                num_images  # Show progress out of target number
            )
            tasks.append(task)
            
            # Stop adding tasks once we have enough
            if len(tasks) >= num_images:
                break
        
        print(f"\nPre-filtered: Skipped {skipped_existing} already processed images")
        
        if len(tasks) == 0:
            print("\nNo new images to process!")
            return {"success": 0, "failed": 0, "skipped": skipped_existing}
        
        print(f"Starting concurrent processing of {len(tasks)} new images...")
        
        # Process with progress bar
        results_list = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing images"):
            result = await coro
            results_list.append(result)
        
        # Aggregate results
        total_results = {"success": 0, "failed": 0, "skipped": skipped_existing}
        for result in results_list:
            total_results["success"] += result["success"]
            total_results["failed"] += result["failed"]
            total_results["skipped"] += result["skipped"]
        
        return total_results


async def main():
    parser = argparse.ArgumentParser(description='Ingest random LAION images to generate embeddings')
    
    parser.add_argument('--fal-endpoint', default=os.getenv('FAL_ENDPOINT'), help='FAL endpoint URL')
    parser.add_argument('--fal-key', default=os.getenv('FAL_KEY'), help='FAL API key')
    parser.add_argument('--lancedb-uri', default=os.getenv('LANCEDB_URI'), help='LanceDB URI')
    parser.add_argument('--lancedb-key', default=os.getenv('LANCEDB_API_KEY'), help='LanceDB API key')
    parser.add_argument('--num-images', type=int, default=1000, help='Number of images to process')
    parser.add_argument('--concurrency', type=int, default=5, help='Number of concurrent requests')
    parser.add_argument('--min-score', type=float, default=5.0, help='Minimum aesthetic score (default: 5.0)')
    
    args = parser.parse_args()
    
    if not all([args.fal_endpoint, args.fal_key, args.lancedb_uri, args.lancedb_key]):
        print("Error: Missing required parameters. Please set environment variables:")
        print("  FAL_ENDPOINT, FAL_KEY, LANCEDB_URI, LANCEDB_API_KEY")
        return
    
    async with LAIONImageIngestionPipeline(
        fal_endpoint=args.fal_endpoint,
        fal_key=args.fal_key,
        lancedb_uri=args.lancedb_uri,
        lancedb_api_key=args.lancedb_key
    ) as pipeline:
        start_time = datetime.now()
        
        results = await pipeline.ingest_laion_dataset(
            num_images=args.num_images,
            concurrency=args.concurrency,
            min_aesthetic_score=args.min_score
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n{'='*50}")
        print("LAION IMAGE INGESTION COMPLETE")
        print(f"{'='*50}")
        print(f"Target Images: {args.num_images}")
        print(f"Min Aesthetic Score: {args.min_score}")
        print(f"Concurrency: {args.concurrency}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Success: {results['success']} embeddings")
        print(f"Failed: {results['failed']} embeddings")
        print(f"Skipped: {results['skipped']} images")
        print(f"Rate: {results['success'] / duration:.2f} embeddings/second" if duration > 0 else "N/A")


if __name__ == "__main__":
    asyncio.run(main())