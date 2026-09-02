"""
Micro-benchmark da latência de **inferência pura** (sem HTTP) dos três backends:

* sklearn `Pipeline` (`models/classifier.pkl`)  — baseline
* ONNX Runtime fp32 (`models/classifier.onnx`)
* ONNX Runtime int8 (`models/classifier.int8.onnx`)  — quantizado

Isola o que a otimização da Etapa 4 realmente afeta. Grava
`benchmarks/inference_comparison.json`.

    python scripts/bench_inference.py --n 2000
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import joblib
import numpy as np

_BASE = Path(__file__).resolve().parents[1]
PATH_PKL = _BASE / "models" / "classifier.pkl"
PATH_ONNX = _BASE / "models" / "classifier.onnx"
PATH_ONNX_INT8 = _BASE / "models" / "classifier.int8.onnx"

SAMPLE = (
    "patient presents with acute chest pain elevated troponin and st segment "
    "elevation on ecg consistent with myocardial infarction history of "
    "hypertension and prior coronary artery disease"
)


def _pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(int(len(s) * p), len(s) - 1)]


def _bench(fn, n: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    lat = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        lat.append((time.perf_counter() - t0) * 1000)
    return {
        "mean_ms": round(statistics.fmean(lat), 4),
        "p50_ms": round(_pct(lat, 0.50), 4),
        "p95_ms": round(_pct(lat, 0.95), 4),
        "p99_ms": round(_pct(lat, 0.99), 4),
        "min_ms": round(min(lat), 4),
        "throughput_rps": round(n / (sum(lat) / 1000), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--out", default="benchmarks/inference_comparison.json")
    args = parser.parse_args()

    import onnxruntime as ort

    pipeline = joblib.load(PATH_PKL)
    sess_fp32 = ort.InferenceSession(str(PATH_ONNX), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(PATH_ONNX_INT8), providers=["CPUExecutionProvider"])
    onnx_in = np.array([SAMPLE], dtype=object).reshape(-1, 1)
    name = sess_fp32.get_inputs()[0].name

    backends = {
        "sklearn_pkl": lambda: (pipeline.predict([SAMPLE]), pipeline.predict_proba([SAMPLE])),
        "onnx_fp32": lambda: sess_fp32.run(None, {name: onnx_in}),
        "onnx_int8": lambda: sess_int8.run(None, {name: onnx_in}),
    }

    result = {"n": args.n, "warmup": args.warmup, "backends": {}}
    for label, fn in backends.items():
        r = _bench(fn, args.n, args.warmup)
        result["backends"][label] = r
        print(f"{label:14s} mean={r['mean_ms']:.3f} ms  p50={r['p50_ms']:.3f}  "
              f"p95={r['p95_ms']:.3f}  p99={r['p99_ms']:.3f}  ({r['throughput_rps']} rps)")

    base = result["backends"]["sklearn_pkl"]["mean_ms"]
    result["speedup_vs_sklearn"] = {
        k: round(base / v["mean_ms"], 2) for k, v in result["backends"].items()
    }
    print("\nspeedup (mean) vs sklearn:", result["speedup_vs_sklearn"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Salvo em: {out}")


if __name__ == "__main__":
    main()
