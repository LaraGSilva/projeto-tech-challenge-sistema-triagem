"""Testes da API de triagem de laudos médicos."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_MODEL = Path(__file__).resolve().parents[1] / "models" / "classifier.pkl"
pytestmark = pytest.mark.skipif(
    not _MODEL.exists(),
    reason="models/classifier.pkl ausente — rode `python -m src.pipeline` antes",
)

LAUDO = (
    "Patient with malignant neoplasm of the lung, metastatic carcinoma "
    "with tumor cells infiltrating adjacent tissue."
)


@pytest.fixture(scope="module")
def client():
    from src.app.api import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_predict_sucesso(client):
    resp = client.post("/predict", json={"texto": LAUDO})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["condition_label"], int)
    assert body["condition_name"]
    assert 0.0 <= body["confianca"] <= 1.0
    assert abs(sum(body["probabilidades"].values()) - 1.0) < 1e-3
    assert body["inference_time_ms"] > 0


def test_predict_texto_ausente(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_predict_texto_sem_conteudo(client):
    resp = client.post("/predict", json={"texto": "12345 !!! ---"})
    assert resp.status_code == 422


def test_metrics_expoe_prometheus(client):
    client.post("/predict", json={"texto": LAUDO})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "api_requests_total" in resp.text
    assert "model_inference_latency_seconds" in resp.text
