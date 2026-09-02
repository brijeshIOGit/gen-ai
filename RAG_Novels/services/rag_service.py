import logging


logger = logging.getLogger("uvicorn.error")


class RAGService:
    def __init__(
        self,
        loader,
        chunker,
        embedding_provider,
        vector_store,
        chat_model,
        reranker=None,
        reranker_candidate_k=20,
    ):
        self.loader = loader
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.chat_model = chat_model
        self.reranker = reranker
        self.reranker_candidate_k = reranker_candidate_k

    def ingest(self, directory_path):
        logger.info("Ingestion started: %s", directory_path)
        raw_documents = self.loader.load(directory_path)
        chunks = self.chunker.chunk_all(raw_documents)

        # keep object association intact at app/service layer
        total_chunks = len(chunks)
        batch_size = 100
        logger.info("Ingestion: generating embeddings for %d chunks in batches of %d", total_chunks, batch_size)
        for start in range(0, total_chunks, batch_size):
            batch = chunks[start:start + batch_size]
            embeddings = self.embedding_provider.embed_many([chunk.text for chunk in batch])
            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding
            completed = min(start + batch_size, total_chunks)
            logger.info("Ingestion: embedded %d/%d chunks", completed, total_chunks)

        # flatten only for vector store API
        ids = [chunk.id for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]

        logger.info("Ingestion: storing %d chunks in the vector store", total_chunks)
        self.vector_store.upsert(ids=ids, documents=texts, embeddings=embeddings)
        logger.info("Ingestion completed: %d chunks stored", total_chunks)

        return chunks

    def ask(self, question, top_k=3):
        logger.info("Question: generating query embedding")
        question_embedding = self.embedding_provider.embed(question)
        candidate_k = max(top_k, self.reranker_candidate_k) if self.reranker else top_k
        result = self.vector_store.query(question_embedding, top_k=candidate_k)

        documents = result.get("documents", [[]])[0]
        if self.reranker:
            documents = self.reranker.rerank(question, documents, top_k=top_k)
        else:
            documents = documents[:top_k]
        context = "\n\n".join(documents)

        answer = self.chat_model.answer(question, context)
        embedding_usage = self.embedding_provider.last_usage
        chat_usage = answer["usage"]
        return {
            "text": answer["text"],
            "usage": {
                "embedding_prompt_tokens": embedding_usage["prompt_tokens"],
                "answer_prompt_tokens": chat_usage["prompt_tokens"],
                "answer_completion_tokens": chat_usage["completion_tokens"],
                "total_tokens": embedding_usage["total_tokens"] + chat_usage["total_tokens"],
            },
        }