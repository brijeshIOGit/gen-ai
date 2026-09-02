from collections.abc import Sequence
from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    def rerank(self, query: str, documents: Sequence[str], top_k: int) -> list[str]: ...


class VectorStore(Protocol):
    def query(self, embedding: list[float], top_k: int = 3) -> dict: ...


class ChatModel(Protocol):
    def answer(self, question: str, context: str) -> dict: ...
