# RAG_Novels/domain/chunker.py or wherever you place it

import logging

from RAG_Novels.domain.chunk import DocumentChunk


logger = logging.getLogger(__name__)


class TextChunker:
    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_all(self, documents):
        chunks = []
        logger.info("Ingestion: chunking %d documents", len(documents))

        for document_index, doc in enumerate(documents, start=1):
            text = doc["content"]
            start = 0
            index = 0

            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end]

                chunks.append(
                    DocumentChunk(
                        id=f"{doc['id']}_chunk{index + 1}",
                        text=chunk_text,
                        source_file=doc["source_file"],
                    )
                )

                start += self.chunk_size - self.overlap
                index += 1

                logger.info("Ingestion: chunked document %d/%d: %s (%d chunks)", document_index, len(documents), doc["source_file"], index)

            logger.info("Ingestion: created %d chunks", len(chunks))
        return chunks