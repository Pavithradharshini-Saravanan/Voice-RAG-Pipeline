import re
import time
import math
import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field

from backend.stt.stt_service import STTResult, stt_service
from backend.vector_db.index import SearchResult, vector_index
from backend.guardrails.guardrail_manager import guardrail_manager, GuardrailResult
from backend.config import settings

logger = logging.getLogger(__name__)

def safe_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except Exception:
        return default

# Structured Pydantic Input/Output Schemas
class HarnessRequest(BaseModel):
    query_text: Optional[str] = None
    audio_base64: Optional[str] = None
    stt_provider: str = Field(default="auto", description="STT Provider: sarvam, elevenlabs, or mock")
    chunking_strategy: str = Field(default="semantic", description="fixed_size, semantic, metadata_aware, hierarchical, recursive")
    sarvam_key: Optional[str] = None
    elevenlabs_key: Optional[str] = None

class ToolCallLog(BaseModel):
    tool_name: str
    status: str
    latency_ms: float
    details: Dict[str, Any] = {}

class PipelineTiming(BaseModel):
    stt_ms: float = 0.0
    guardrail_input_ms: float = 0.0
    vector_retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    guardrail_output_ms: float = 0.0
    total_latency_ms: float = 0.0
    target_met: bool = False

class HarnessResponse(BaseModel):
    transcription: str
    stt_provider: str
    answer: str
    refused: bool = False
    refusal_reason: Optional[str] = None
    retrieved_chunks: List[Dict[str, Any]] = []
    chunking_strategy: str
    grounding_score: float
    timing: PipelineTiming
    tool_calls: List[ToolCallLog] = []

class AgentHarness:
    """Structured Agent Orchestration Harness with tool calls, retry policies, schema validation, and error recovery."""
    
    def __init__(self, max_retries: int = 2, backoff_factor: float = 0.5):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    async def _execute_with_retry(self, func: Callable, *args, **kwargs):
        """Executes async function with exponential backoff retry policy."""
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(f"Harness Attempt {attempt + 1} failed ({e}). Retrying...")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.backoff_factor * (2 ** attempt))
        raise last_exception or Exception("Execution failed after retries.")

    def _generate_low_latency_answer(self, query: str, chunks: List[SearchResult]) -> str:
        """Synthesizes rich, exact answers directly matching query intent and facts from retrieved passage context."""
        if not chunks:
            return "No relevant context found in the dataset to answer your question."

        top_chunk = chunks[0].chunk
        doc_title = top_chunk.metadata.get("title") or "MSMARCO Passage"
        text = top_chunk.text.strip()

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences or len(sentences) == 1:
            return f"Based on {doc_title}: {text}"

        query_words = set(w.lower() for w in re.findall(r'\w+', query) if len(w) >= 3)
        
        # Stopwords to filter out generic terms
        stopwords = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how", "does", "about", "tell", "show", "give", "list"}
        specific_query_words = query_words - stopwords

        if not specific_query_words:
            specific_query_words = query_words

        # Score each sentence based on specific query word matches
        sentence_scores = []
        for idx, sentence in enumerate(sentences):
            s_words = set(w.lower() for w in re.findall(r'\w+', sentence))
            # Calculate match count
            match_score = sum(1.5 if word in s_words else 0 for word in specific_query_words)
            # Penalize generic definition sentence if specific action/detail keywords match later sentences
            if idx == 0 and ("is a" in sentence.lower() or "is the" in sentence.lower()) and len(specific_query_words) > 1:
                match_score *= 0.7
            sentence_scores.append((match_score, idx, sentence))

        # Sort by match score descending
        sentence_scores.sort(key=lambda x: x[0], reverse=True)

        best_score, best_idx, best_sentence = sentence_scores[0]

        # If query asks for details, list, or missions, combine relevant detailed sentences
        if len(sentences) > 1 and best_score > 0:
            if best_idx != 0:
                answer_body = f"{best_sentence}"
            else:
                answer_body = " ".join(sentences[:2])
        else:
            answer_body = text

        answer = f"Based on {doc_title}: {answer_body}"
        if not answer.endswith("."):
            answer += "."

        return answer

    async def process_pipeline(
        self,
        request: HarnessRequest,
        audio_bytes: Optional[bytes] = None
    ) -> HarnessResponse:
        """Runs full structured agent orchestration pipeline."""
        t_start = time.perf_counter()
        tool_logs: List[ToolCallLog] = []
        timing = PipelineTiming()

        # -------------------------------------------------------------
        # STEP 1: Speech-To-Text Transcription (Tool Call 1)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        if audio_bytes:
            stt_res: STTResult = await self._execute_with_retry(
                stt_service.transcribe,
                audio_bytes=audio_bytes,
                provider=request.stt_provider,
                simulated_text=request.query_text,
                sarvam_key=request.sarvam_key or "",
                elevenlabs_key=request.elevenlabs_key or ""
            )
            query_text = stt_res.text
            provider_used = stt_res.provider
            stt_ms = stt_res.latency_ms
        else:
            query_text = request.query_text or "What is photosynthesis?"
            provider_used = "text_direct"
            stt_ms = (time.perf_counter() - t0) * 1000.0

        timing.stt_ms = round(stt_ms, 2)
        tool_logs.append(ToolCallLog(
            tool_name="speech_to_text_tool",
            status="success",
            latency_ms=timing.stt_ms,
            details={"provider": provider_used, "transcription": query_text}
        ))

        # -------------------------------------------------------------
        # STEP 2: Input Safety & Guardrail Check (Tool Call 2)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        is_safe, safety_reason = guardrail_manager.check_input_safety(query_text)
        guardrail_in_ms = (time.perf_counter() - t0) * 1000.0
        timing.guardrail_input_ms = round(guardrail_in_ms, 2)

        if not is_safe:
            tool_logs.append(ToolCallLog(
                tool_name="input_guardrail_tool",
                status="refusal",
                latency_ms=timing.guardrail_input_ms,
                details={"reason": safety_reason}
            ))
            total_ms = (time.perf_counter() - t_start) * 1000.0
            timing.total_latency_ms = round(total_ms, 2)
            timing.target_met = timing.total_latency_ms <= settings.LATENCY_TARGET_MS
            return HarnessResponse(
                transcription=query_text,
                stt_provider=provider_used,
                answer="I cannot fulfill this request as it contains unsafe or prohibited content.",
                refused=True,
                refusal_reason=safety_reason,
                chunking_strategy=request.chunking_strategy,
                grounding_score=0.0,
                timing=timing,
                tool_calls=tool_logs
            )

        tool_logs.append(ToolCallLog(
            tool_name="input_guardrail_tool",
            status="passed",
            latency_ms=timing.guardrail_input_ms,
            details={"safe": True}
        ))

        # -------------------------------------------------------------
        # STEP 3: Vector DB Retrieval (Tool Call 3)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        search_results, search_ms = vector_index.search(
            query=query_text,
            strategy=request.chunking_strategy,
            top_k=settings.TOP_K_RETRIEVAL
        )
        timing.vector_retrieval_ms = round(search_ms, 2)

        formatted_chunks = [
            {
                "chunk_id": r.chunk.chunk_id,
                "doc_id": r.chunk.doc_id,
                "score": round(safe_float(r.score), 4),
                "text": r.chunk.text,
                "strategy": r.chunk.strategy,
                "metadata": r.chunk.metadata
            } for r in search_results
        ]

        tool_logs.append(ToolCallLog(
            tool_name="vector_db_retrieval_tool",
            status="success",
            latency_ms=timing.vector_retrieval_ms,
            details={"strategy": request.chunking_strategy, "chunks_found": len(search_results)}
        ))

        # -------------------------------------------------------------
        # STEP 4: Answer Generation (Tool Call 4)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        candidate_answer = self._generate_low_latency_answer(query_text, search_results)
        gen_ms = (time.perf_counter() - t0) * 1000.0
        timing.generation_ms = round(gen_ms, 2)

        tool_logs.append(ToolCallLog(
            tool_name="answer_generation_tool",
            status="success",
            latency_ms=timing.generation_ms,
            details={"answer_length": len(candidate_answer)}
        ))

        # -------------------------------------------------------------
        # STEP 5: Grounding & Hallucination Guardrail Check (Tool Call 5)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        g_res: GuardrailResult = guardrail_manager.validate(query_text, search_results, candidate_answer)
        g_out_ms = (time.perf_counter() - t0) * 1000.0
        timing.guardrail_output_ms = round(g_out_ms, 2)

        refused = not g_res.passed_all
        final_answer = candidate_answer
        if refused:
            final_answer = f"I am unable to answer this question. {g_res.refusal_reason or 'Context relevance low.'}"

        g_score_safe = round(safe_float(g_res.grounding_score), 2)
        tool_logs.append(ToolCallLog(
            tool_name="grounding_guardrail_tool",
            status="passed" if g_res.passed_all else "refused",
            latency_ms=timing.guardrail_output_ms,
            details={
                "grounding_score": g_score_safe,
                "is_grounded": g_res.is_grounded,
                "is_on_topic": g_res.is_on_topic
            }
        ))

        # Final timing calculation
        total_ms = (time.perf_counter() - t_start) * 1000.0
        timing.total_latency_ms = round(total_ms, 2)
        timing.target_met = timing.total_latency_ms <= settings.LATENCY_TARGET_MS

        return HarnessResponse(
            transcription=query_text,
            stt_provider=provider_used,
            answer=final_answer,
            refused=refused,
            refusal_reason=g_res.refusal_reason,
            retrieved_chunks=formatted_chunks,
            chunking_strategy=request.chunking_strategy,
            grounding_score=g_score_safe,
            timing=timing,
            tool_calls=tool_logs
        )

agent_harness = AgentHarness()
