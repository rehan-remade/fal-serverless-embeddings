from typing import Optional, List
from pydantic import BaseModel, Field


class VLM2VecEmbeddingRequest(BaseModel):
    """Request model for VLM2Vec embedding generation"""
    text: Optional[str] = Field(
        None, 
        description="Text input for embedding generation"
    )
    image_url: Optional[str] = Field(
        None, 
        description="URL of image for embedding generation"
    )
    video_url: Optional[str] = Field(
        None, 
        description="URL of video for embedding generation"
    )
    max_pixels: int = Field(
        360 * 420, 
        description="Maximum pixels for video processing"
    )
    fps: float = Field(
        1.0, 
        description="FPS for video processing"
    )


class VLM2VecEmbeddingResponse(BaseModel):
    """Response model for VLM2Vec embedding generation"""
    embedding: List[float] = Field(
        description="Generated embedding vector"
    )
    dimension: int = Field(
        description="Embedding dimension"
    )