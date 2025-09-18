
This restructured approach provides:

1. **Clean Architecture**: Separates concerns between app logic, worker logic, and utilities
2. **No Local Module Issues**: The VLM2Vec repo is cloned during deployment, avoiding local/remote mismatch
3. **Production Pattern**: Follows the same pattern as the HuMo example
4. **Distributed Support**: Ready for multi-GPU deployment if needed
5. **Better Error Handling**: Comprehensive error handling throughout
6. **Type Safety**: Full type annotations for better code quality

The key improvements:
- Uses a worker pattern for model inference
- Clones VLM2Vec repo during deployment
- Separates request/response models
- Has proper constants and utilities
- Follows fal best practices for production apps

# VLM2Vec Fal Serverless Deployment

This directory contains a properly structured fal serverless deployment for VLM2Vec-V2.0.

## Architecture

The app follows a modular architecture similar to production fal apps:

- **app.py**: Main fal app with endpoints and request handling
- **worker.py**: Distributed worker for model inference
- **models.py**: Pydantic models for request/response
- **constants.py**: Configuration constants
- **utils.py**: Utility functions and distributed runner
- **deploy.py**: Deployment automation script

## Features

- **Clean separation of concerns**: App logic, worker logic, and utilities are separated
- **Distributed processing support**: Ready for multi-GPU inference
- **Proper error handling**: Comprehensive error handling throughout
- **Type safety**: Full type annotations
- **Production-ready**: Follows fal best practices

## Setup

1. **Install dependencies**:
   ```bash
   pip install fal-client
   ```

2. **Run setup script** (optional, for local development):
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Deploy the app**:
   ```bash
   python deploy.py
   ```

   Or for development mode:
   ```bash
   python deploy.py --dev
   ```

## Usage

### Generate Text Embedding

```python
import fal_client

result = await fal_client.submit_async(
    "fal-ai/vlm2vec-v2/embed",
    arguments={
        "text": "A beautiful sunset over the ocean"
    }
)
embedding = result["embedding"]
```

### Generate Image Embedding

```python
result = await fal_client.submit_async(
    "fal-ai/vlm2vec-v2/embed",
    arguments={
        "image_url": "https://example.com/image.jpg"
    }
)
embedding = result["embedding"]
```

### Generate Video Embedding

```python
result = await fal_client.submit_async(
    "fal-ai/vlm2vec-v2/embed",
    arguments={
        "video_url": "https://example.com/video.mp4",
        "fps": 1.0,
        "max_pixels": 360 * 420
    }
)
embedding = result["embedding"]
```

## Integration with LanceDB

```python
import lancedb
import fal_client

# Initialize LanceDB
db = lancedb.connect("./lancedb")
table = db.create_table("embeddings", {
    "id": str,
    "content": str,
    "embedding": list,
    "metadata": dict,
})

# Generate and store embedding
result = await fal_client.submit_async(
    "fal-ai/vlm2vec-v2/embed",
    arguments={"text": "Sample text"}
)

table.add([{
    "id": "unique-id",
    "content": "Sample text",
    "embedding": result["embedding"],
    "metadata": {"type": "text"}
}])

# Search
query_result = await fal_client.submit_async(
    "fal-ai/vlm2vec-v2/embed",
    arguments={"text": "Query text"}
)
results = table.search(query_result["embedding"]).limit(5).to_list()
```

## Development

The codebase is structured for easy extension:

1. **Add new endpoints**: Edit `app.py`
2. **Modify model logic**: Edit `worker.py`
3. **Add new request types**: Edit `models.py`
4. **Change configurations**: Edit `constants.py`

## Troubleshooting

- **Import errors**: Ensure the VLM2Vec repo is properly cloned
- **GPU memory**: Model requires ~6GB VRAM
- **Network timeouts**: Check file download limits in `constants.py`


