import re
from typing import List, Dict, Any
from backend.dataset_loader import Document
from backend.chunking.base import BaseChunker, Chunk

class FixedSizeChunker(BaseChunker):
    """1. Fixed-Size Chunking with configurable sliding window overlap."""
    def __init__(self, chunk_size: int = 150, overlap: int = 40):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document: Document) -> List[Chunk]:
        chunks = []
        text = document.text
        text_len = len(text)
        if text_len == 0:
            return chunks

        start = 0
        chunk_idx = 0
        step = max(1, self.chunk_size - self.overlap)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end]
            chunk_id = f"{document.doc_id}_fixed_{chunk_idx}"
            
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                text=chunk_text,
                start_char=start,
                end_char=end,
                strategy="fixed_size",
                metadata={"chunk_size": self.chunk_size, "overlap": self.overlap, "title": document.title}
            ))
            
            if end >= text_len:
                break
            start += step
            chunk_idx += 1

        return chunks


class SemanticChunker(BaseChunker):
    """2. Semantic / Sentence-Boundary Chunking splitting at natural linguistic/sentence shifts."""
    def __init__(self, target_sentences: int = 2):
        self.target_sentences = target_sentences

    def chunk_document(self, document: Document) -> List[Chunk]:
        chunks = []
        # Split into sentences using regex boundary detection
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', document.text) if s.strip()]
        if not sentences:
            return chunks

        chunk_idx = 0
        current_char = 0

        for i in range(0, len(sentences), self.target_sentences):
            group = sentences[i : i + self.target_sentences]
            chunk_text = " ".join(group)
            start_char = document.text.find(chunk_text, current_char)
            if start_char == -1:
                start_char = current_char
            end_char = start_char + len(chunk_text)
            current_char = end_char
            
            chunk_id = f"{document.doc_id}_semantic_{chunk_idx}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
                strategy="semantic",
                metadata={"sentences_count": len(group), "title": document.title, "language": document.language}
            ))
            chunk_idx += 1

        return chunks


class MetadataAwareChunker(BaseChunker):
    """3. Metadata-Preserving Chunking: injects document title, doc_id, and structural tags into every chunk context."""
    def __init__(self, chunk_size: int = 180, overlap: int = 30):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document: Document) -> List[Chunk]:
        chunks = []
        raw_text = document.text
        text_len = len(raw_text)
        if text_len == 0:
            return chunks

        header = f"[Doc: {document.title or document.doc_id} | Lang: {document.language}] "
        start = 0
        chunk_idx = 0
        step = max(1, self.chunk_size - self.overlap)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            body = raw_text[start:end]
            full_text = f"{header}{body}"
            chunk_id = f"{document.doc_id}_meta_{chunk_idx}"

            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                text=full_text,
                start_char=start,
                end_char=end,
                strategy="metadata_aware",
                metadata={
                    "title": document.title,
                    "language": document.language,
                    "url": document.url,
                    "has_header": True
                }
            ))

            if end >= text_len:
                break
            start += step
            chunk_idx += 1

        return chunks


class HierarchicalChunker(BaseChunker):
    """4. Hierarchical (Parent-Child) Chunking: small child chunks for vector indexing mapping to full parent passage."""
    def __init__(self, child_size: int = 90):
        self.child_size = child_size

    def chunk_document(self, document: Document) -> List[Chunk]:
        chunks = []
        # Parent chunk covers full document passage
        parent_id = f"{document.doc_id}_parent"
        parent_chunk = Chunk(
            chunk_id=parent_id,
            doc_id=document.doc_id,
            text=document.text,
            start_char=0,
            end_char=len(document.text),
            strategy="hierarchical_parent",
            metadata={"role": "parent", "title": document.title}
        )
        chunks.append(parent_chunk)

        # Child chunks
        text = document.text
        text_len = len(text)
        start = 0
        child_idx = 0

        while start < text_len:
            end = min(start + self.child_size, text_len)
            child_text = text[start:end]
            child_id = f"{document.doc_id}_child_{child_idx}"

            chunks.append(Chunk(
                chunk_id=child_id,
                doc_id=document.doc_id,
                text=child_text,
                start_char=start,
                end_char=end,
                strategy="hierarchical_child",
                parent_chunk_id=parent_id,
                metadata={"role": "child", "parent_id": parent_id, "title": document.title}
            ))

            if end >= text_len:
                break
            start += self.child_size
            child_idx += 1

        return chunks


class RecursiveCharacterChunker(BaseChunker):
    """5. Recursive Separator Chunking: splits text recursively on multi-level delimiters."""
    def __init__(self, max_chunk_size: int = 160, separators: List[str] = None):
        self.max_chunk_size = max_chunk_size
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        if not text:
            return []
        if len(text) <= self.max_chunk_size or not separators:
            return [text]

        sep = separators[0]
        splits = text.split(sep) if sep else list(text)
        
        final_chunks = []
        current = ""
        
        for s in splits:
            item = s + sep if sep else s
            if len(current) + len(item) <= self.max_chunk_size:
                current += item
            else:
                if current.strip():
                    final_chunks.append(current.strip())
                if len(item) > self.max_chunk_size and len(separators) > 1:
                    final_chunks.extend(self._split_text(item, separators[1:]))
                    current = ""
                else:
                    current = item

        if current.strip():
            final_chunks.append(current.strip())
        return final_chunks

    def chunk_document(self, document: Document) -> List[Chunk]:
        chunks = []
        split_texts = self._split_text(document.text, self.separators)

        current_pos = 0
        for idx, t in enumerate(split_texts):
            start = document.text.find(t, current_pos)
            if start == -1:
                start = current_pos
            end = start + len(t)
            current_pos = end

            chunks.append(Chunk(
                chunk_id=f"{document.doc_id}_recursive_{idx}",
                doc_id=document.doc_id,
                text=t,
                start_char=start,
                end_char=end,
                strategy="recursive",
                metadata={"title": document.title}
            ))

        return chunks


def get_chunker(strategy_name: str) -> BaseChunker:
    """Factory function for chunking strategies."""
    strategies = {
        "fixed_size": FixedSizeChunker,
        "semantic": SemanticChunker,
        "metadata_aware": MetadataAwareChunker,
        "hierarchical": HierarchicalChunker,
        "recursive": RecursiveCharacterChunker
    }
    chunker_cls = strategies.get(strategy_name, SemanticChunker)
    return chunker_cls()
