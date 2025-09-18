"""Utility functions and classes for VLM2Vec fal app"""

import asyncio
import os
import tempfile
import aiohttp
from typing import Any, Type
from multiprocessing import Process, Queue
import torch
import torch.distributed as dist


class DistributedWorker:
    """Base class for distributed workers"""
    
    def __init__(self, rank: int, world_size: int, device: str = "cuda"):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}" if device == "cuda" else device)
    
    def setup(self, **kwargs):
        """Setup method to be implemented by subclasses"""
        raise NotImplementedError
    
    def __call__(self, **kwargs):
        """Call method to be implemented by subclasses"""
        raise NotImplementedError


class DistributedRunner:
    """Runner for distributed processing"""
    
    def __init__(self, worker_cls: Type[DistributedWorker], world_size: int = 1, cwd: str = None):
        self.worker_cls = worker_cls
        self.world_size = world_size
        self.cwd = cwd
        self.processes = []
        self.queues = []
        print(f"[DistributedRunner] Initialized with worker_cls={worker_cls.__name__}, world_size={world_size}, cwd={cwd}")
    
    async def start(self, **setup_kwargs):
        """Start worker processes"""
        print(f"[DistributedRunner] Starting {self.world_size} worker processes")
        for rank in range(self.world_size):
            queue = Queue()
            self.queues.append(queue)
            
            process = Process(
                target=self._worker_loop,
                args=(rank, queue, setup_kwargs)
            )
            process.start()
            self.processes.append(process)
            print(f"[DistributedRunner] Started process for rank {rank}, PID: {process.pid}")
        
        # Wait for all workers to be ready
        print("[DistributedRunner] Waiting for workers to initialize...")
        await asyncio.sleep(2)
        print("[DistributedRunner] Workers should be ready")
    
    def _worker_loop(self, rank: int, queue: Queue, setup_kwargs: dict):
        """Worker process loop"""
        try:
            print(f"[Worker {rank}] Starting worker loop")
            
            if self.cwd:
                os.chdir(self.cwd)
                print(f"[Worker {rank}] Changed directory to: {self.cwd}")
            
            # Initialize distributed if world_size > 1
            if self.world_size > 1:
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12355'
                print(f"[Worker {rank}] Initializing distributed group")
                dist.init_process_group(backend='nccl', rank=rank, world_size=self.world_size)
            
            # Create and setup worker
            print(f"[Worker {rank}] Creating worker instance")
            worker = self.worker_cls(rank, self.world_size)
            
            print(f"[Worker {rank}] Calling worker setup with kwargs: {list(setup_kwargs.keys())}")
            worker.setup(**setup_kwargs)
            print(f"[Worker {rank}] Worker setup complete")
            
            # Process requests
            while True:
                kwargs = queue.get()
                if kwargs is None:
                    print(f"[Worker {rank}] Received None, exiting")
                    break
                
                print(f"[Worker {rank}] Processing request")
                result = worker(**kwargs)
                print(f"[Worker {rank}] Request processed, returning result")
                queue.put(result)
                
        except Exception as e:
            import traceback
            error_msg = f"[Worker {rank}] Fatal error: {str(e)}"
            print(error_msg)
            print(f"[Worker {rank}] Traceback:\n{traceback.format_exc()}")
            # Put error in queue so parent process knows
            queue.put({"error": error_msg, "traceback": traceback.format_exc()})
    
    async def invoke(self, kwargs: dict) -> dict:
        """Invoke worker with given arguments"""
        print(f"[DistributedRunner] Invoking with kwargs: {list(kwargs.keys())}")
        
        # Send work to rank 0
        self.queues[0].put(kwargs)
        print("[DistributedRunner] Sent work to rank 0")
        
        # Collect results
        results = []
        for i, queue in enumerate(self.queues):
            print(f"[DistributedRunner] Waiting for result from queue {i}")
            result = await asyncio.get_event_loop().run_in_executor(None, queue.get)
            print(f"[DistributedRunner] Got result from queue {i}: {list(result.keys()) if result else 'None'}")
            if result:
                results.append(result)
        
        # Return first non-empty result
        final_result = results[0] if results else {}
        print(f"[DistributedRunner] Returning final result: {list(final_result.keys()) if final_result else 'None'}")
        return final_result
    
    def __del__(self):
        """Cleanup processes"""
        for queue in self.queues:
            queue.put(None)
        
        for process in self.processes:
            process.terminate()
            process.join()


async def download_file_to_dir_async(url: str, output_dir: str, max_size: int = None) -> str:
    """Download file from URL to directory asynchronously"""
    import uuid
    
    # Determine file extension from URL
    ext = os.path.splitext(url)[1] or '.tmp'
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(output_dir, filename)
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response.raise_for_status()
            
            # Check file size if max_size is specified
            if max_size and response.headers.get('Content-Length'):
                content_length = int(response.headers['Content-Length'])
                if content_length > max_size:
                    raise ValueError(f"File size {content_length} exceeds maximum {max_size}")
            
            # Download file
            with open(filepath, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)
    
    return filepath


def get_seed(seed: int = None) -> int:
    """Get seed value, generate random if not provided"""
    import random
    import sys
    
    if seed is None or seed < 0:
        return random.randint(0, sys.maxsize)
    return seed

