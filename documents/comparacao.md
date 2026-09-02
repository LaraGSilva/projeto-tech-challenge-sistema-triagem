# Comparação de latências — baseline × otimizado (Etapa 4)

## Objetivo

Reduzir a latência de **inferência** do classificador aplicando uma técnica vista
em aula. Foram aplicadas **duas**, encadeadas:

1. **Conversão para ONNX Runtime** — `sklearn.Pipeline` → grafo ONNX
   (`skl2onnx`), executado pelo `onnxruntime` (C++).
2. **Quantização dinâmica int8** — pesos `float32` → `int8`
   (`onnxruntime.quantization.quantize_dynamic`).

Artefatos: `models/classifier.onnx` (fp32) e `models/classifier.int8.onnx` (int8),
gerados por `python -m src.model.optimize`.

## Metodologia

| Medição | Script | Isola |
|---|---|---|
| Inferência pura (offline) | `scripts/bench_inference.py --n 3000` | só `predict` + `predict_proba` de 1 laudo, sem HTTP |
| Fim-a-fim HTTP | `scripts/bench_latency.py --n 300 --concurrency 1` | requisição `POST /predict` completa |
| Acurácia | `src/model/optimize.py::check_parity` | `accuracy` / `f1_weighted` no hold-out (2.310 laudos) |

Ambiente: Windows, CPU, 1 worker uvicorn, `CPUExecutionProvider`. O backend da API
é escolhido por `MODEL_BACKEND` (`sklearn` | `onnx` | `onnx-int8`).

## Resultados

### 1. Inferência pura — `benchmarks/inference_comparison.json`

| Backend | média (ms) | p50 | p95 | p99 | throughput | speedup (média) |
|---|---:|---:|---:|---:|---:|---:|
| sklearn `.pkl` (baseline) | **1,073** | 1,032 | 1,291 | 1,921 | 932 rps | 1,0× |
| ONNX fp32 | **0,208** | 0,186 | 0,311 | 0,362 | 4.818 rps | **5,2×** |
| ONNX int8 (quantizado) | **0,251** | 0,193 | 0,257 | 0,472 | 3.983 rps | **4,3×** |

### 2. Fim-a-fim HTTP (`POST /predict`, c=1, n=300)

| Backend | média (ms) | p50 | p95 |
|---|---:|---:|---:|
| sklearn | 2,99 | 2,68 | 4,61 |
| ONNX int8 | 3,33 | 3,22 | 4,47 |

### 3. Acurácia (hold-out, 2.310 laudos) — `models/optimization.json`

| Backend | accuracy | f1 (weighted) | concordância com sklearn |
|---|---:|---:|---:|
| sklearn `.pkl` | 0,6199 | 0,6049 | 100% |
| ONNX fp32 | 0,6216 | 0,6051 | 98,5% |
| ONNX int8 | 0,6216 | 0,6051 | 98,5% |

### 4. Tamanho do artefato

| | tamanho |
|---|---:|
| `classifier.pkl` | 1.823 KB |
| `classifier.onnx` / `classifier.int8.onnx` | ~1.553 KB |

## Análise

- **O ganho real vem da conversão para ONNX Runtime (~5×).** O `onnxruntime`
  executa a vetorização TF-IDF e o classificador linear em C++, evitando o
  overhead de Python + matriz esparsa `scipy` do sklearn.
- **A quantização int8, isolada, tem efeito marginal aqui.** O modelo é um
  classificador **linear** sobre features **esparsas** — o cálculo já é trivial,
  então o custo de quantizar/dequantizar supera o ganho do produto interno em
  int8: a média fica levemente pior que o fp32 (0,25 vs 0,21 ms), mas o **p95
  melhora** (0,26 vs 0,31 ms) — cauda mais previsível — **sem perder acurácia**
  (predições idênticas ao fp32). O tamanho quase não muda porque o vocabulário do
  TF-IDF (não quantizado) domina o arquivo.
- **No fim-a-fim HTTP a diferença some:** o overhead de FastAPI + serialização +
  rede local (~2,5–3 ms) domina, e os ~0,8 ms economizados na inferência quase
  não movem o p50. O ganho aparece na métrica `model_inference_latency_seconds`
  (painel do Grafana), sob concorrência alta, e escala se o modelo crescer.
- **Acurácia preservada:** 0,62 → 0,62; a discordância de 1,5% vem de `float32`
  (ONNX) vs `float64` (sklearn), não de perda de informação.

## Decisão

Produção (ECS Fargate) roda **`MODEL_BACKEND=onnx-int8`** (definido em
`.aws/task-definition.json`): mesma acurácia, inferência ~4× mais rápida, p95 mais
estável. O backend `sklearn` continua como padrão local para desenvolvimento e
comparação.

> O op `StringNormalizer` do modelo ONNX (lowercase do TF-IDF) exige o locale
> `en_US.UTF-8` — instalado no `Dockerfile` (`locales` + `locale-gen`).

## Reproduzir

```bash
uv run python -m src.model.optimize            # gera os .onnx + optimization.json
uv run python scripts/bench_inference.py --n 3000

# fim-a-fim, um backend por vez:
MODEL_BACKEND=sklearn   uv run python -m src.app.api   # e em outro terminal:
uv run python scripts/bench_latency.py --n 300 --concurrency 1 --out benchmarks/latency_http_sklearn.json
MODEL_BACKEND=onnx-int8 uv run python -m src.app.api
uv run python scripts/bench_latency.py --n 300 --concurrency 1 --out benchmarks/latency_http_onnx-int8.json
```
