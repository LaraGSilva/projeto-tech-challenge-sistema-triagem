# =============================================================================
# Imagem de inferência — API de Triagem de Laudos Médicos (FastAPI + sklearn/ONNX)
# Backend escolhido por MODEL_BACKEND: sklearn | onnx | onnx-int8 (ver Etapa 4).
# =============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    MODEL_PATH=/app/models/classifier.pkl \
    METADATA_PATH=/app/models/metadata.json

# locale en_US.UTF-8 — exigido pelo op StringNormalizer do modelo ONNX
RUN apt-get update && apt-get install -y --no-install-recommends locales \
    && sed -i '/^# *en_US.UTF-8 UTF-8/s/^# *//' /etc/locale.gen \
    && locale-gen \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Dependências (camada cacheável)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 2) Código + artefatos do modelo (.pkl + .onnx)
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

# 1 worker de propósito: latência determinística
CMD ["uvicorn", "src.app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
