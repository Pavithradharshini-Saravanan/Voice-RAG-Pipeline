from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any
from backend.dataset_loader import Document

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_chunk_id: str = ""

class BaseChunker(ABC):
    @abstractmethod
    def chunk_document(self, document: Document) -> List[Chunk]:
        pass

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
