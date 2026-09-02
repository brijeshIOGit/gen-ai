from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DocumentChunk:
    id: str
    text: str
    source_file: str
    embedding: Optional[List[float]] = None