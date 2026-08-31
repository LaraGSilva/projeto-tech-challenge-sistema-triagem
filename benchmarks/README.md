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
- A **Etapa 4** (ONNX Runtime / quantização) ataca a parte de inferência —
  a comparação deve usar `model_inference_latency_seconds` e o mesmo script,
  gerando `benchmarks/latency_optimized.json` para o "antes x depois".
- Ambiente da medição: Docker Desktop (Windows), 1 worker uvicorn. Registrar
  a máquina usada ao gravar o vídeo.

Arquivos gerados: `latency_baseline.json`, `latency_baseline_c10.json`.
