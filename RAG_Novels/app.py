import os

from dotenv import load_dotenv
import chromadb
from fastapi import FastAPI
from openai import OpenAI

from RAG_Novels.config import AppConfig

try:
    from RAG_Novels.adapters.chat.openai_chat_model import OpenAIChatModel
    from RAG_Novels.adapters.embeddings.openai_embedding_provider import OpenAIEmbeddingProvider
    from RAG_Novels.adapters.vectorstores.chroma_vector_store import ChromaVectorStore
    from RAG_Novels.domain.chunker import TextChunker
    from RAG_Novels.loaders.text_loader import TextFileLoader
    from RAG_Novels.routes import router
    from RAG_Novels.services.rag_service import RAGService
except ModuleNotFoundError:
    from adapters.chat.openai_chat_model import OpenAIChatModel
    from adapters.embeddings.openai_embedding_provider import OpenAIEmbeddingProvider
    from adapters.vectorstores.chroma_vector_store import ChromaVectorStore
    from domain.chunker import TextChunker
    from loaders.text_loader import TextFileLoader
    from routes import router
    from services.rag_service import RAGService

load_dotenv(os.path.expanduser("~/.zshrc"))


def build_rag_service():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in ~/.zshrc")

    config = AppConfig.from_environment()
    client = OpenAI()
    chroma_client = chromadb.PersistentClient(path=config.chroma_path)
    collection = chroma_client.get_or_create_collection(name=config.collection_name)

    loader = TextFileLoader()
    chunker = TextChunker(chunk_size=config.chunk_size, overlap=config.chunk_overlap)
    embedding_provider = OpenAIEmbeddingProvider(client, config.embedding_model)
    vector_store = ChromaVectorStore(collection=collection)
    chat_model = OpenAIChatModel(client, config.chat_model)

    reranker = None
    if config.reranker_enabled:
        from RAG_Novels.adapters.rerankers.sentence_transformers_reranker import SentenceTransformersReranker

        reranker = SentenceTransformersReranker(config.reranker_model, config.reranker_device)

    return RAGService(
        loader=loader,
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        chat_model=chat_model,
        reranker=reranker,
        reranker_candidate_k=config.reranker_candidate_k,
    )


app = FastAPI(title="RAG Novels API", version="1.0.0")
app.state.rag_service = build_rag_service()


app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("RAG_Novels.app:app", host="0.0.0.0", port=8000, reload=True)