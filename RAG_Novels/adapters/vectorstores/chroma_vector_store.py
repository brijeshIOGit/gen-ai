import logging


logger = logging.getLogger("uvicorn.error")


class ChromaVectorStore:
    def __init__(self, client=None, collection_name=None, collection=None):
        if collection is not None:
            self.collection = collection
        elif client is not None and collection_name is not None:
            self.collection = client.get_or_create_collection(name=collection_name)
        else:
            raise ValueError("Provide either a Chroma collection or both client and collection_name.")

    def upsert(self, ids, documents, embeddings):
        batch_size = 5_000
        total_items = len(ids)
        for start in range(0, total_items, batch_size):
            end = min(start + batch_size, total_items)
            self.collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                embeddings=embeddings[start:end],
            )
            logger.info("Ingestion: stored %d/%d chunks", end, total_items)

    def query(self, embedding, top_k=3):
        return self.collection.query(query_embeddings=[embedding], n_results=top_k)