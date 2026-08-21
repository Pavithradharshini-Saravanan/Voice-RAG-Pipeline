import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import asyncio
import time
import json
import logging
from backend.vector_db.index import vector_index
from backend.analytics.benchmark import benchmark_suite, BENCHMARK_TEST_QUERIES

logging.basicConfig(level=logging.INFO)

async def main():
    print("=" * 70)
    print("      HH GOA 2026 VOICE-ENABLED RAG PIPELINE BENCHMARK HARNESS     ")
    print("=" * 70)

    # 1. Initialize Index
    print("\n[1/3] Initializing Vector Index across 5 Vast Chunking Strategies...")
    t0 = time.perf_counter()
    vector_index.initialize_index()
    init_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[OK] Index initialized in {init_ms:.2f}ms for {len(vector_index.documents)} MSMARCO-XI documents.")

    # 2. Run Benchmark across Chunking Strategies
    strategies = ["semantic", "fixed_size", "metadata_aware", "hierarchical", "recursive"]
    benchmark_reports = {}

    print(f"\n[2/3] Executing Latency Benchmark Suite ({len(BENCHMARK_TEST_QUERIES)} test queries)...")
    
    for strat in strategies:
        print(f"\n---> Benchmarking Strategy: '{strat.upper()}'")
        report = await benchmark_suite.run_benchmark(strategy=strat, stt_provider="mock")
        benchmark_reports[strat] = report
        
        m = report["latency_metrics"]
        print(f"      P50 Latency (Median):  {m['p50_ms']} ms")
        print(f"      P70 Latency:           {m['p70_ms']} ms")
        print(f"      P90 Latency:           {m['p90_ms']} ms")
        print(f"      P100 Latency (Max):    {m['p100_ms']} ms")
        print(f"      Target < 200ms Met:    {'YES [OK]' if report['target_200ms_met'] else 'NO [X]'}")
        print(f"      Phase Breakdown:       {json.dumps(m['phase_breakdown'])}")

    # 3. Overall Summary Table
    print("\n" + "=" * 70)
    print("                      LATENCY ANALYTICS SUMMARY TABLE               ")
    print("=" * 70)
    print(f"{'Strategy':<18} | {'P50 (ms)':<9} | {'P70 (ms)':<9} | {'P90 (ms)':<9} | {'P100 (ms)':<10} | {'Target <200ms'}")
    print("-" * 70)
    for strat, rep in benchmark_reports.items():
        m = rep["latency_metrics"]
        status = "PASSED (<200ms)" if rep["target_200ms_met"] else "EXCEEDED"
        print(f"{strat:<18} | {m['p50_ms']:<9} | {m['p70_ms']:<9} | {m['p90_ms']:<9} | {m['p100_ms']:<10} | {status}")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
