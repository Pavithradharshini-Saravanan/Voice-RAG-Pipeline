import os
from pathlib import Path
from dataclasses import dataclass, field

BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class Settings:
    BASE_DIR: Path = BASE_DIR
    
    # API Keys
    SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "")
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # STT Defaults
    DEFAULT_STT_PROVIDER: str = os.getenv("DEFAULT_STT_PROVIDER", "mock")  # "sarvam", "elevenlabs", "mock"
    SARVAM_STT_URL: str = "https://api.sarvam.ai/speech-to-text"
    ELEVENLABS_STT_URL: str = "https://api.elevenlabs.io/v1/speech-to-text"
    
    # RAG & Embedding Settings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    VECTOR_DIM: int = 384
    TOP_K_RETRIEVAL: int = 3
    SIMILARITY_THRESHOLD: float = 0.30
    
    # Latency Target & Benchmarking
    LATENCY_TARGET_MS: float = 200.0
    BENCHMARK_SAMPLE_COUNT: int = 50
    
    # Dataset Settings
    DATASET_NAME: str = "ai4bharat/MSMARCO-XI"
    DATASET_MAX_PASSAGES: int = 500
    DATASET_CACHE_DIR: Path = field(default_factory=lambda: BASE_DIR / "data_cache")

settings = Settings()
