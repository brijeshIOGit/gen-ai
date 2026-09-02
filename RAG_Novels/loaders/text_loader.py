import os
import logging


logger = logging.getLogger(__name__)


class TextFileLoader:
    def load(self, directory_path: str):
        if not directory_path:
            raise FileNotFoundError("No directory path provided.")

        if not os.path.isdir(directory_path):
            raise FileNotFoundError(f"Directory does not exist: {directory_path}")

        filenames = [filename for filename in sorted(os.listdir(directory_path)) if filename.endswith(".txt")]
        logger.info("Ingestion: reading %d text files from %s", len(filenames), directory_path)
        documents = []
        for index, filename in enumerate(filenames, start=1):
            path = os.path.join(directory_path, filename)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append({
                "id": filename,
                "source_file": filename,
                "content": content
            })
            logger.info("Ingestion: read file %d/%d: %s (%d characters)", index, len(filenames), filename, len(content))

        logger.info("Ingestion: finished reading %d files", len(documents))
        return documents


def load_text_files(directory_path: str):
    return TextFileLoader().load(directory_path)