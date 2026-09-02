from dataclasses import dataclass
import os
from pathlib import Path


@dataclass
class AppConfig:
    data_dir: str = str(Path(__file__).resolve().parent / "novels")
    chroma_path: str = "./chroma_persistent_storage"
    collection_name: str = "document_novel_collection"
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 3
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: str = "cpu"
    reranker_candidate_k: int = 20

    @classmethod
    def from_environment(cls):
        return cls(
            reranker_enabled=os.getenv("RERANKER_ENABLED", "false").lower() in {"1", "true", "yes"},
            reranker_model=os.getenv("RERANKER_MODEL", cls.reranker_model),
            reranker_device=os.getenv("RERANKER_DEVICE", cls.reranker_device),
            reranker_candidate_k=int(os.getenv("RERANKER_CANDIDATE_K", cls.reranker_candidate_k)),
        )