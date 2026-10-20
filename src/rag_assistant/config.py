from functools import lru_cache
from pathlib import Path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)
    app_env: str = "development"
    log_level: str = "INFO"
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen3.5:4b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_num_ctx: int = 8192
    ollama_temperature: float = 0.1
    ollama_num_predict: int = 1024
    ollama_timeout_seconds: float = 180
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "research_chunks"
    data_dir: Path = Path("./data")
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 120
    retrieval_top_k: int = 5
    retrieval_candidates: int = 20
    rrf_k: int = 60
    max_upload_mb: int = 25
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    phoenix_enabled: bool = False
    phoenix_endpoint: str = "http://localhost:6006"

    @model_validator(mode="after")
    def validate_local_provider(self):
        if self.llm_provider.lower() != "ollama":
            raise ValueError("LLM_PROVIDER must be 'ollama'; hosted inference providers are not supported.")
        if self.chunk_size_tokens <= 0 or not 0 <= self.chunk_overlap_tokens < self.chunk_size_tokens:
            raise ValueError("Chunk overlap must be non-negative and smaller than chunk size.")
        if self.ollama_num_ctx < 512 or self.max_upload_mb < 1:
            raise ValueError("OLLAMA_NUM_CTX must be >=512 and MAX_UPLOAD_MB must be positive.")
        if self.ollama_timeout_seconds <= 0 or not 0 <= self.ollama_temperature <= 2:
            raise ValueError("OLLAMA_TIMEOUT_SECONDS must be positive and OLLAMA_TEMPERATURE must be between 0 and 2.")
        if self.retrieval_top_k < 1 or self.retrieval_candidates < self.retrieval_top_k or self.rrf_k < 1:
            raise ValueError("Retrieval top-k/candidate counts and RRF_K must be positive; candidates must cover top-k.")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
