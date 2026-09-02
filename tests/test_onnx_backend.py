"""Testes do backend de inferência ONNX (Etapa 4)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_MODELS = Path(__file__).resolve().parents[1] / "models"
pytestmark = pytest.mark.skipif(
    not (_MODELS / "classifier.int8.onnx").exists(),
    reason="models/classifier.int8.onnx ausente — rode `python -m src.model.optimize`",
)

LAUDO = (
    "Patient with malignant neoplasm of the lung, metastatic carcinoma "
    "with tumor cells infiltrating adjacent tissue."
)


@pytest.fixture(params=["onnx", "onnx-int8"])
def onnx_client(request, monkeypatch):
    monkeypatch.setenv("MODEL_BACKEND", request.param)
    from src.app.api import app

    with TestClient(app) as c:
        c.backend = request.param
        yield c


def test_backend_reportado(onnx_client):
    assert onnx_client.get("/health").json()["backend"] == onnx_client.backend


def test_predict_onnx(onnx_client):
    resp = onnx_client.post("/predict", json={"texto": LAUDO})
    assert resp.status_code == 200
    body = resp.json()
    assert body["condition_label"] in (1, 2, 3, 4, 5)
    assert abs(sum(body["probabilidades"].values()) - 1.0) < 1e-3


def test_predict_sklearn_e_onnx_concordam():
    """A classe prevista deve ser a mesma nos dois backends para um laudo claro."""
    import joblib

    from src.model.onnx_infer import OnnxClassifier
    from src.model.text import clean_text

    texto = clean_text(LAUDO)
    sk = joblib.load(_MODELS / "classifier.pkl")
    onnx = OnnxClassifier(_MODELS / "classifier.int8.onnx", classes=[1, 2, 3, 4, 5])
    assert int(sk.predict([texto])[0]) == int(onnx.predict([texto])[0])
