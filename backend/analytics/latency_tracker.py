import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class LatencyMetrics:
    sample_count: int
    p50_ms: float
    p70_ms: float
    p90_ms: float
    p100_ms: float  # Maximum latency
    mean_ms: float
    min_ms: float
    phase_breakdown: Dict[str, float] = field(default_factory=dict)
    target_met_ratio: float = 0.0

class LatencyTracker:
    """Calculates P50, P70, P90, and P100 latency statistics across pipeline runs."""
    
    def calculate_percentiles(self, latencies_ms: List[float], phase_data: Optional[List[Dict[str, float]]] = None) -> LatencyMetrics:
        if not latencies_ms:
            return LatencyMetrics(
                sample_count=0, p50_ms=0.0, p70_ms=0.0, p90_ms=0.0, p100_ms=0.0,
                mean_ms=0.0, min_ms=0.0
            )

        arr = np.array(latencies_ms)
        p50 = float(np.percentile(arr, 50))
        p70 = float(np.percentile(arr, 70))
        p90 = float(np.percentile(arr, 90))
        p100 = float(np.max(arr))
        mean_val = float(np.mean(arr))
        min_val = float(np.min(arr))

        # Target met ratio (< 200ms)
        target_met = float(np.sum(arr <= 200.0) / len(arr))

        # Calculate average breakdown per phase if provided
        phase_averages = {}
        if phase_data:
            keys = phase_data[0].keys()
            for key in keys:
                vals = [p[key] for p in phase_data if key in p]
                if vals:
                    phase_averages[key] = round(float(np.mean(vals)), 2)

        return LatencyMetrics(
            sample_count=len(latencies_ms),
            p50_ms=round(p50, 2),
            p70_ms=round(p70, 2),
            p90_ms=round(p90, 2),
            p100_ms=round(p100, 2),
            mean_ms=round(mean_val, 2),
            min_ms=round(min_val, 2),
            phase_breakdown=phase_averages,
            target_met_ratio=round(target_met, 4)
        )

latency_tracker = LatencyTracker()
