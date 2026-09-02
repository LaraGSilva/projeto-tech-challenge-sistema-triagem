"""
Etapa 4 — otimização de latência de inferência.

Converte o `Pipeline` sklearn (`models/classifier.pkl`) para **ONNX Runtime** e
aplica **quantização dinâmica int8**. Gera:

* `models/classifier.onnx`       — ONNX fp32
* `models/classifier.int8.onnx`  — ONNX com pesos quantizados (int8)

e verifica a paridade de predições contra o `.pkl` no hold-out
(`models/test_split.pkl`).

    python -m src.model.optimize
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parents[2]
PATH_PKL = _BASE / "models" / "classifier.pkl"
PATH_ONNX = _BASE / "models" / "classifier.onnx"
PATH_ONNX_INT8 = _BASE / "models" / "classifier.int8.onnx"
PATH_TEST_SPLIT = _BASE / "models" / "test_split.pkl"
PATH_REPORT = _BASE / "models" / "optimization.json"

TARGET_OPSET = 17


def convert_to_onnx(pkl_path: Path = PATH_PKL, onnx_path: Path = PATH_ONNX) -> Path:
    """Converte o Pipeline (TfidfVectorizer + LogisticRegression) para ONNX fp32."""
    pipeline = joblib.load(pkl_path)
    logger.info("Convertendo %s para ONNX (opset %d)...", pkl_path.name, TARGET_OPSET)
    onnx_model = to_onnx(
        pipeline,
        initial_types=[("input", StringTensorType([None, 1]))],
        options={id(pipeline): {"zipmap": False}},  # saída = array de probabilidades
        target_opset=TARGET_OPSET,
    )
    onnx_path.write_bytes(onnx_model.SerializeToString())
    logger.info("ONNX fp32: %s (%d KB)", onnx_path, onnx_path.stat().st_size // 1024)
    return onnx_path


def quantize_int8(
    onnx_path: Path = PATH_ONNX, int8_path: Path = PATH_ONNX_INT8
) -> Path:
    """Quantização dinâmica: pesos fp32 -> int8 (onnxruntime.quantization)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    logger.info("Quantização dinâmica int8 de %s...", onnx_path.name)
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
    logger.info("ONNX int8: %s (%d KB)", int8_path, int8_path.stat().st_size // 1024)
    return int8_path


def _metrics(y_true, y_pred, ref_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "agreement_sklearn": round(float((np.asarray(y_pred) == np.asarray(ref_pred)).mean()), 4),
    }


def check_parity() -> dict:
    """Compara accuracy/F1 do .pkl vs ONNX fp32 vs ONNX int8 no hold-out."""
    import onnxruntime as ort

    pipeline = joblib.load(PATH_PKL)
    data = joblib.load(PATH_TEST_SPLIT)
    x_test = list(data["X_test"])
    y_test = np.asarray(data["y_test"])
    x_arr = np.array(x_test, dtype=object).reshape(-1, 1)

    def onnx_pred(path: Path):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return sess.run(None, {sess.get_inputs()[0].name: x_arr})[0]

    sk_pred = pipeline.predict(x_test)
    report = {
        "n_holdout": len(x_test),
        "sklearn_pkl": _metrics(y_test, sk_pred, sk_pred),
        "onnx_fp32": _metrics(y_test, onnx_pred(PATH_ONNX), sk_pred),
        "onnx_int8": _metrics(y_test, onnx_pred(PATH_ONNX_INT8), sk_pred),
    }
    report["sizes_kb"] = {
        "sklearn_pkl": PATH_PKL.stat().st_size // 1024,
        "onnx_fp32": PATH_ONNX.stat().st_size // 1024,
        "onnx_int8": PATH_ONNX_INT8.stat().st_size // 1024,
    }
    PATH_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("\n%s", json.dumps(report, indent=2))
    return report


def run() -> dict:
    convert_to_onnx()
    quantize_int8()
    return check_parity()


if __name__ == "__main__":
    run()
