"""
API REST de triagem de laudos médicos.

Recebe o texto de um laudo e retorna a categoria de doença prevista pelo
classificador NLP (``models/classifier.pkl`` — Pipeline TF-IDF + LogisticRegression).

Instrumentada com ``prometheus_client``:
  * ``api_requests_total``              — contagem de requisições (método, rota, status)
  * ``api_request_latency_seconds``     — latência HTTP fim-a-fim
  * ``model_inference_latency_seconds`` — latência só da inferência do modelo
  * ``model_predictions_total``         — predições por classe

Endpoints:
  POST /predict   — classifica um laudo
  GET  /health    — liveness/readiness
  GET  /metrics   — métricas no formato Prometheus
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from src.model.text import clean_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(_BASE / "models" / "classifier.pkl")))
METADATA_PATH = Path(os.getenv("METADATA_PATH", str(_BASE / "models" / "metadata.json")))
INDEX_HTML = Path(__file__).parent / "index.html"

# Backend de inferência: "sklearn" (default) | "onnx" (ONNX fp32) | "onnx-int8"
# (ONNX quantizado). Lido em tempo de startup. Ver Etapa 4 / documents/comparacao.md.
ONNX_PATH = _BASE / "models" / "classifier.onnx"
ONNX_INT8_PATH = _BASE / "models" / "classifier.int8.onnx"


def _resolve_backend() -> str:
    return os.getenv("MODEL_BACKEND", "sklearn").lower()

# ── Métricas Prometheus ──────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total de requisições HTTP",
    ["method", "endpoint", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latência HTTP fim-a-fim em segundos",
    ["method", "endpoint"],
)
INFERENCE_LATENCY = Histogram(
    "model_inference_latency_seconds",
    "Latência somente da inferência do modelo em segundos",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total de predições por classe",
    ["condition_label", "condition_name"],
)

_ml: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo uma única vez, na subida do processo."""
    metadata = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    _ml["metadata"] = metadata
    _ml["label_names"] = {
        int(k): str(v) for k, v in metadata.get("label_names", {}).items()
    }

    backend = _resolve_backend()
    if backend in ("onnx", "onnx-int8"):
        from src.model.onnx_infer import OnnxClassifier

        onnx_file = ONNX_INT8_PATH if backend == "onnx-int8" else ONNX_PATH
        logger.info("Carregando modelo ONNX (%s): %s", backend, onnx_file)
        _ml["model"] = OnnxClassifier(onnx_file, classes=metadata.get("classes"))
    else:
        logger.info("Carregando modelo sklearn: %s", MODEL_PATH)
        _ml["model"] = joblib.load(MODEL_PATH)

    _ml["backend"] = backend
    logger.info(
        "Modelo carregado | backend=%s | classes=%s",
        backend,
        list(_ml["model"].classes_),
    )
    yield
    _ml.clear()


app = FastAPI(
    title="API de Triagem de Laudos Médicos",
    description="Classificação NLP da categoria de doença a partir do texto do laudo",
    version="1.0.0",
    lifespan=lifespan,
)

# API pública sem auth — libera CORS para o frontend funcionar mesmo servido
# de outra origem (arquivo local, Live Server, etc.).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Contratos ────────────────────────────────────────────────────────────────
class LaudoRequest(BaseModel):
    texto: str = Field(
        ...,
        min_length=1,
        description="Texto livre do laudo/abstract médico",
        examples=[
            "Patient presents with acute chest pain, elevated troponin and "
            "ST-segment elevation on ECG consistent with myocardial infarction."
        ],
    )


class PredicaoResponse(BaseModel):
    condition_label: int
    condition_name: str
    confianca: float
    probabilidades: dict[str, float]
    inference_time_ms: float


# ── Instrumentação ───────────────────────────────────────────────────────────
@app.middleware("http")
async def prometheus_middleware(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start

    endpoint = request.url.path
    REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
    REQUEST_LATENCY.labels(request.method, endpoint).observe(elapsed)
    return response


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def index():
    """Frontend simples de teste da API."""
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "model" in _ml,
        "backend": _ml.get("backend", _resolve_backend()),
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredicaoResponse)
def predict(payload: LaudoRequest):
    model = _ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado")

    texto_limpo = clean_text(payload.texto)
    if not texto_limpo:
        raise HTTPException(
            status_code=422, detail="Texto sem conteúdo útil após a limpeza"
        )

    start = time.perf_counter()
    pred = int(model.predict([texto_limpo])[0])
    proba = model.predict_proba([texto_limpo])[0]
    inference_s = time.perf_counter() - start
    INFERENCE_LATENCY.observe(inference_s)

    label_names = _ml.get("label_names", {})
    classes = [int(c) for c in model.classes_]
    probabilidades = {
        label_names.get(c, str(c)): round(float(p), 4)
        for c, p in zip(classes, proba)
    }
    condition_name = label_names.get(pred, str(pred))
    PREDICTION_COUNT.labels(str(pred), condition_name).inc()

    return PredicaoResponse(
        condition_label=pred,
        condition_name=condition_name,
        confianca=round(float(proba.max()), 4),
        probabilidades=probabilidades,
        inference_time_ms=round(inference_s * 1000, 3),
    )


if __name__ == "__main__":
    import uvicorn

    # passa o objeto `app` (não a string) para não reimportar este módulo e
    # registrar as métricas Prometheus duas vezes (DuplicateTimeseries).
    uvicorn.run(app, host="0.0.0.0", port=8000)
