# Baseline de latência — API de Triagem (antes da Etapa 4)

Medições do modelo **sem otimização** (`Pipeline` TF-IDF + LogisticRegression,
`joblib`), servido por FastAPI/uvicorn (1 worker) em container Docker.

Como reproduzir:

```bash
docker compose -f docker-compose.monitoring.yaml up --build -d
python scripts/bench_latency.py --n 300  --concurrency 1
python scripts/bench_latency.py --n 600  --concurrency 10 --out benchmarks/latency_baseline_c10.json
```

## Resultado (referência)

| Cenário            | p50     | p95     | p99      | média   | throughput |
|--------------------|---------|---------|----------|---------|------------|
| Sequencial (c=1)   | ~51 ms  | ~55 ms  | ~193 ms  | ~53 ms  | ~19 req/s  |
| Concorrente (c=10) | ~56 ms  | ~80 ms  | ~114 ms  | ~58 ms  | ~170 req/s |

**Inferência pura do modelo** (métrica `model_inference_latency_seconds`):
`sum/count ≈ 2,6 ms` por predição.

## Leitura

- A inferência do modelo em si já é barata (~2–3 ms). O restante da latência
  fim-a-fim (~45–50 ms no ambiente local via Docker Desktop) é overhead de
  HTTP / framework / port-forward do Docker no Windows.
- A **Etapa 4** (ONNX Runtime + quantização int8) ataca a parte de inferência.
  Resultado: inferência pura de **1,07 ms → 0,21 ms (ONNX) → 0,25 ms (int8)**,
  ~4–5× mais rápida, sem perda de acurácia. Comparação completa em
  [../documents/comparacao.md](../documents/comparacao.md).
- Ambiente da medição: 1 worker uvicorn, CPU. Registrar a máquina no vídeo.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `latency_baseline.json`, `latency_baseline_c10.json` | baseline HTTP (antes da Etapa 4) |
| `inference_comparison.json` | inferência pura: sklearn × ONNX fp32 × ONNX int8 (`scripts/bench_inference.py`) |
| `latency_http_sklearn.json`, `latency_http_onnx-int8.json` | fim-a-fim HTTP por backend |
