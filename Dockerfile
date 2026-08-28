# =============================================================================
# Imagem de inferência — API de Triagem de Laudos Médicos (FastAPI + sklearn)
# Baseline SEM otimização (Etapa 4 aplica ONNX/quantização depois).
# =============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/classifier.pkl \
    METADATA_PATH=/app/models/metadata.json

WORKDIR /app

# 1) Dependências (camada cacheável)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 2) Código + artefato do modelo já treinado
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

# 1 worker de propósito: baseline de latência determinístico
CMD ["uvicorn", "src.app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
