from typing import Literal
from pydantic import BaseModel, Field


Strategy = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    title: str | None = None
    page: int | None = None
    position: int
    text: str
    content_hash: str
    score: float = 0.0
    metadata: dict = Field(default_factory=dict)


class SourceReference(BaseModel):
    marker: int
    document_id: str
    filename: str
    title: str | None = None
    page: int | None = None
    chunk_id: str
    excerpt: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    strategy: Strategy = "hybrid"
    top_k: int = Field(default=5, ge=1, le=30)
    rerank: bool = False


class QueryResponse(BaseModel):
    answer: str
    citations: list[SourceReference]
    retrieved: list[Chunk]
    strategy: str
    latency_ms: dict[str, float]


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    title: str | None = None
    chunks: int
    status: str
    content_hash: str


class HealthResponse(BaseModel):
    status: str
    ollama: bool
    qdrant: bool
    detail: str
