import os
import time
import logging
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

from backend.config import settings
from backend.vector_db.index import vector_index
from backend.harness.agent_harness import agent_harness, HarnessRequest, HarnessResponse
from backend.analytics.benchmark import benchmark_suite
from backend.dataset_loader import dataset_loader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("VoiceRAG-App")

app = FastAPI(
    title="HH Goa 2026 Voice-Enabled RAG Pipeline",
    description="Low-latency voice RAG with Sarvam/ElevenLabs STT, vast chunking, P50/P70/P100 analytics & guardrails",
    version="1.0.0"
)

# Enable CORS for local testing & Web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event: pre-initialize vector index
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Voice-Enabled RAG Index...")
    docs = dataset_loader.load_dataset()
    vector_index.initialize_index(documents=docs)
    logger.info("RAG Index startup complete.")

# Static Files serving
frontend_path = settings.BASE_DIR / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_file = frontend_path / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h2>Voice-Enabled RAG API is running. UI loading...</h2>")

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "vector_index_ready": vector_index._is_initialized,
        "documents_count": len(vector_index.documents),
        "target_latency_ms": settings.LATENCY_TARGET_MS
    }

class VoiceRAGFormRequest(BaseModel):
    query_text: Optional[str] = None
    stt_provider: str = "auto"
    chunking_strategy: str = "semantic"
    sarvam_key: Optional[str] = None
    elevenlabs_key: Optional[str] = None

@app.post("/api/voice-rag", response_model=HarnessResponse)
async def voice_rag_endpoint(
    audio: Optional[UploadFile] = File(None),
    query_text: Optional[str] = Form(None),
    stt_provider: str = Form("auto"),
    chunking_strategy: str = Form("semantic"),
    sarvam_key: Optional[str] = Form(None),
    elevenlabs_key: Optional[str] = Form(None)
):
    """Primary Voice-Enabled RAG Pipeline endpoint handling audio or text inputs."""
    audio_bytes = None
    if audio:
        audio_bytes = await audio.read()

    req = HarnessRequest(
        query_text=query_text,
        stt_provider=stt_provider,
        chunking_strategy=chunking_strategy,
        sarvam_key=sarvam_key,
        elevenlabs_key=elevenlabs_key
    )

    try:
        response = await agent_harness.process_pipeline(req, audio_bytes=audio_bytes)
        return response
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class BenchmarkRequest(BaseModel):
    chunking_strategy: str = "semantic"
    stt_provider: str = "mock"

@app.post("/api/benchmark")
async def run_benchmark_endpoint(req: BenchmarkRequest):
    """Triggers automated test suite benchmark calculating P50, P70, P100 latency percentiles."""
    try:
        summary = await benchmark_suite.run_benchmark(
            strategy=req.chunking_strategy,
            stt_provider=req.stt_provider
        )
        return summary
    except Exception as e:
        logger.error(f"Benchmark error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chunk-comparison")
async def chunk_comparison_endpoint(query: str = "What is photosynthesis?", top_k: int = 2):
    """Compares retrieval performance and chunking layout across all 5 chunking strategies."""
    strategies = ["fixed_size", "semantic", "metadata_aware", "hierarchical", "recursive"]
    comparison = {}

    for strat in strategies:
        results, search_ms = vector_index.search(query=query, strategy=strat, top_k=top_k)
        comparison[strat] = {
            "strategy": strat,
            "retrieval_ms": round(search_ms, 2),
            "top_chunks": [
                {
                    "chunk_id": r.chunk.chunk_id,
                    "score": round(r.score, 4),
                    "text": r.chunk.text,
                    "metadata": r.chunk.metadata
                } for r in results
            ]
        }

    return {"query": query, "comparison": comparison}
