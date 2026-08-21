import re
import time
import math
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from backend.vector_db.index import SearchResult
from backend.config import settings

logger = logging.getLogger(__name__)

# List of prohibited / unsafe patterns
UNSAFE_PATTERNS = [
    r"ignore previous instructions",
    r"bypass system",
    r"system prompt",
    r"execute order",
    r"hack into",
    r"malware",
    r"exploit"
]

@dataclass
class GuardrailResult:
    is_safe: bool
    is_on_topic: bool
    is_grounded: bool
    passed_all: bool
    grounding_score: float
    refusal_reason: Optional[str] = None
    execution_time_ms: float = 0.0

class MultiLayerGuardrailManager:
    """Comprehensive Guardrail System: Input Safety, Off-Topic Filter, Grounding & Hallucination Checks."""
    
    def check_input_safety(self, query: str) -> Tuple[bool, Optional[str]]:
        """1. Input Safety & Anti-Jailbreak Guardrail."""
        lower_query = query.lower()
        for pattern in UNSAFE_PATTERNS:
            if re.search(pattern, lower_query):
                return False, "Input contained prohibited security or prompt injection pattern."
        return True, None

    def check_off_topic(self, query: str, search_results: List[SearchResult]) -> Tuple[bool, Optional[str]]:
        """2. Off-Topic & Domain Relevance Guardrail."""
        if not search_results:
            return False, "No relevant context found in the dataset to answer this question."
        
        best_score = search_results[0].score
        threshold = getattr(settings, "SIMILARITY_THRESHOLD", 0.18)
        if best_score < threshold:
            return False, f"Question is not covered in the dataset knowledge base (relevance score {best_score:.2f} < threshold {threshold:.2f})."
        
        return True, None

    def check_grounding(self, answer: str, context_chunks: List[SearchResult]) -> Tuple[bool, float, Optional[str]]:
        """3. Grounding & Hallucination Guardrail: verifies answer content matches retrieved passages."""
        if not context_chunks or not answer:
            return False, 0.0, "Empty answer or missing context chunks."

        combined_context = " ".join([r.chunk.text.lower() for r in context_chunks])
        answer_words = [w.lower() for w in re.findall(r'\w+', answer) if len(w) > 3]
        
        if not answer_words:
            return True, 1.0, None

        # Calculate word overlap ratio with context
        matched_words = sum(1 for word in answer_words if word in combined_context)
        overlap_ratio = matched_words / len(answer_words)

        if overlap_ratio < 0.30:
            return False, overlap_ratio, f"Answer is not sufficiently grounded in retrieved context (grounding ratio {overlap_ratio:.2f} < 0.30)."

        return True, overlap_ratio, None

    def validate(self, query: str, search_results: List[SearchResult], candidate_answer: str = "") -> GuardrailResult:
        """Runs end-to-end guardrail check suite."""
        t0 = time.perf_counter()
        
        # 1. Safety Check
        is_safe, safety_reason = self.check_input_safety(query)
        if not is_safe:
            ms = (time.perf_counter() - t0) * 1000.0
            return GuardrailResult(
                is_safe=False,
                is_on_topic=False,
                is_grounded=False,
                passed_all=False,
                grounding_score=0.0,
                refusal_reason=safety_reason,
                execution_time_ms=ms
            )

        # 2. Off-Topic Check
        is_topic, topic_reason = self.check_off_topic(query, search_results)
        if not is_topic:
            ms = (time.perf_counter() - t0) * 1000.0
            return GuardrailResult(
                is_safe=True,
                is_on_topic=False,
                is_grounded=False,
                passed_all=False,
                grounding_score=0.0,
                refusal_reason=topic_reason,
                execution_time_ms=ms
            )

        # 3. Grounding Check (if answer is provided)
        is_grounded = True
        g_score = 1.0
        g_reason = None
        if candidate_answer:
            is_grounded, g_score, g_reason = self.check_grounding(candidate_answer, search_results)

        ms = (time.perf_counter() - t0) * 1000.0
        passed_all = is_safe and is_topic and is_grounded
        refusal = g_reason if not is_grounded else None

        if math.isnan(g_score) or math.isinf(g_score):
            g_score = 0.0

        return GuardrailResult(
            is_safe=is_safe,
            is_on_topic=is_topic,
            is_grounded=is_grounded,
            passed_all=passed_all,
            grounding_score=g_score,
            refusal_reason=refusal,
            execution_time_ms=ms
        )

guardrail_manager = MultiLayerGuardrailManager()
