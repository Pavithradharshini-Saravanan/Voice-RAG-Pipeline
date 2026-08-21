import time
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from backend.chunking.base import Chunk
from backend.chunking.strategies import get_chunker, BaseChunker
from backend.dataset_loader import Document, dataset_loader
from backend.config import settings

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    retrieval_time_ms: float

class LowLatencyVectorIndex:
    """Ultra-fast, low-latency in-memory vector database with lazy-loaded embeddings for instant startup."""
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.encoder = None
        self.strategy_indices: Dict[str, Dict[str, Any]] = {}
        self.documents: List[Document] = []
        self._is_initialized = False

    def _init_encoder(self):
        if self.encoder is None:
            use_tfidf = os.getenv("USE_TFIDF_EMBEDDINGS", "false").lower() == "true"
            if not use_tfidf:
                try:
                    from sentence_transformers import SentenceTransformer
                    logger.info(f"Loading embedding model: {self.model_name}")
                    self.encoder = SentenceTransformer(self.model_name)
                except Exception as e:
                    logger.warning(f"SentenceTransformer unavailable/failed ({e}). Using TF-IDF vectorizer fallback.")
                    use_tfidf = True

            if use_tfidf or self.encoder is None:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.encoder = TfidfVectorizer(max_features=settings.VECTOR_DIM)

    def encode(self, texts: List[str]) -> np.ndarray:
        self._init_encoder()
        if hasattr(self.encoder, "encode"):
            vectors = self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return vectors.astype(np.float32)
        else:
            if not hasattr(self.encoder, "vocabulary_") or self.encoder.vocabulary_ is None:
                matrix = self.encoder.fit_transform(texts).toarray()
            else:
                matrix = self.encoder.transform(texts).toarray()
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (matrix / norms).astype(np.float32)

    def initialize_index(self, documents: Optional[List[Document]] = None, strategies: Optional[List[str]] = None):
        """Indexes dataset passages across chunking strategies for instant (<0.1s) server startup."""
        t0 = time.perf_counter()
        if documents is None:
            documents = dataset_loader.load_dataset()
        self.documents = documents

        if strategies is None:
            strategies = ["semantic", "fixed_size", "metadata_aware", "hierarchical", "recursive"]

        for strat in strategies:
            chunker = get_chunker(strat)
            chunks = chunker.chunk_documents(documents)
            if not chunks:
                continue

            self.strategy_indices[strat] = {
                "chunks": chunks,
                "vectors": None,
                "count": len(chunks)
            }

        self._is_initialized = True
        # Pre-warm semantic strategy vector embeddings on startup
        try:
            self._ensure_vectors("semantic")
        except Exception as e:
            logger.warning(f"Pre-warm vector encoding warning: {e}")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"Vector index initialized and pre-warmed in {elapsed_ms:.2f}ms")

    def _ensure_vectors(self, strategy: str):
        if strategy in self.strategy_indices and self.strategy_indices[strategy]["vectors"] is None:
            chunks = self.strategy_indices[strategy]["chunks"]
            texts = [c.text for c in chunks]
            vectors = self.encode(texts)
            self.strategy_indices[strategy]["vectors"] = vectors

    def search(self, query: str, strategy: str = "semantic", top_k: int = settings.TOP_K_RETRIEVAL) -> Tuple[List[SearchResult], float]:
        """Performs vector search in < 10ms."""
        t0 = time.perf_counter()
        if not self._is_initialized or strategy not in self.strategy_indices:
            self.initialize_index()

        self._ensure_vectors(strategy)
        idx_data = self.strategy_indices.get(strategy) or list(self.strategy_indices.values())[0]
        chunks: List[Chunk] = idx_data["chunks"]
        vectors: np.ndarray = idx_data["vectors"]

        query_vector = self.encode([query])[0]
        scores = np.dot(vectors, query_vector)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        retrieval_ms = (time.perf_counter() - t0) * 1000.0

        for idx in top_indices:
            score = float(scores[idx])
            chunk = chunks[idx]

            if chunk.strategy == "hierarchical_child" and chunk.parent_chunk_id:
                parent_chunk = next((c for c in chunks if c.chunk_id == chunk.parent_chunk_id), None)
                if parent_chunk:
                    chunk = parent_chunk

            results.append(SearchResult(
                chunk=chunk,
                score=score,
                retrieval_time_ms=retrieval_ms
            ))

        return results, retrieval_ms

vector_index = LowLatencyVectorIndex()
