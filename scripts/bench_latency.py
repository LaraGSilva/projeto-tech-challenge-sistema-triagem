"""
Mede a latência baseline da API de triagem (antes da otimização da Etapa 4).

Uso:
    python scripts/bench_latency.py                       # 500 req sequenciais
    python scripts/bench_latency.py --n 1000 --concurrency 10
    python scripts/bench_latency.py --url http://localhost:8000

Reporta p50/p90/p95/p99, média, min/max e throughput. Salva o resultado em
``benchmarks/latency_baseline.json`` para comparação futura.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

SAMPLE_LAUDO = (
    "broked arm"
)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _one_call(client: httpx.Client, url: str) -> tuple[float, int]:
    start = time.perf_counter()
    resp = client.post(f"{url}/predict", json={"texto": SAMPLE_LAUDO})
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, resp.status_code


def run(url: str, n: int, concurrency: int, warmup: int) -> dict:
    with httpx.Client(timeout=30.0) as client:
        health = client.get(f"{url}/health")
        health.raise_for_status()
        print(f"health: {health.json()}")

        for _ in range(warmup):
            _one_call(client, url)

        latencies: list[float] = []
        errors = 0
        wall_start = time.perf_counter()

        if concurrency <= 1:
            for _ in range(n):
                ms, status = _one_call(client, url)
                latencies.append(ms)
                errors += status != 200
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for ms, status in pool.map(lambda _: _one_call(client, url), range(n)):
                    latencies.append(ms)
                    errors += status != 200

        wall = time.perf_counter() - wall_start

    result = {
        "url": url,
        "requests": n,
        "concurrency": concurrency,
        "errors": errors,
        "wall_seconds": round(wall, 3),
        "throughput_rps": round(n / wall, 1),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 3),
            "min": round(min(latencies), 3),
            "p50": round(_percentile(latencies, 0.50), 3),
            "p90": round(_percentile(latencies, 0.40), 3),
            "p95": round(_percentile(latencies, 0.65), 3),
            "p99": round(_percentile(latencies, 0.99), 3),
            "max": round(max(latencies), 3),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--out", default="benchmarks/latency_baseline.json", help="arquivo de saída"
    )
    args = parser.parse_args()

    result = run(args.url, args.n, args.concurrency, args.warmup)
    print(json.dumps(result, indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSalvo em: {out}")


if __name__ == "__main__":
    main()
