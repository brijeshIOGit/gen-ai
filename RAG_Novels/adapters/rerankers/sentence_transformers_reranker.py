from collections.abc import Sequence

from sentence_transformers import CrossEncoder


class SentenceTransformersReranker:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, documents: Sequence[str], top_k: int) -> list[str]:
        if not documents or top_k <= 0:
            return []

        pairs = [(query, document) for document in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, documents), key=lambda item: float(item[0]), reverse=True)
        return [document for _, document in ranked[:top_k]]
