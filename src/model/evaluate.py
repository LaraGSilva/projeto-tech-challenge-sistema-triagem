"""
Avaliação do classificador NLP de laudos médicos.

Carrega o pipeline treinado e o split de teste persistido por ``train.py``,
calcula accuracy / F1 e o classification report, salva o resultado em
``models/evaluation.json`` e interrompe a pipeline se a accuracy ficar abaixo
do threshold (gate de qualidade para o retreino no Airflow).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parents[2]
PATH_MODEL = _BASE / "models" / "classifier.pkl"
PATH_TEST_SPLIT = _BASE / "models" / "test_split.pkl"
PATH_REPORT = _BASE / "models" / "evaluation.json"

# Baseline clássico (TF-IDF) neste corpus fica em ~0.60 de accuracy.
ACCURACY_THRESHOLD = 0.55


def run_evaluation(
    model_path: str | Path = PATH_MODEL,
    test_path: str | Path = PATH_TEST_SPLIT,
    threshold: float = ACCURACY_THRESHOLD,
) -> dict:
    """Avalia o modelo no conjunto de teste. Levanta ``ValueError`` se reprovar."""
    logger.info("Carregando modelo de: %s", model_path)
    model = joblib.load(model_path)

    logger.info("Carregando split de teste de: %s", test_path)
    data = joblib.load(test_path)
    X_test, y_test = data["X_test"], data["y_test"]

    y_pred = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))
    report_text = classification_report(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    logger.info("\n%s", report_text)
    logger.info("Accuracy: %.4f | F1 (weighted): %.4f", accuracy, f1)

    result = {
        "accuracy": accuracy,
        "f1_weighted": f1,
        "threshold": threshold,
        "model_path": str(model_path),
    }
    PATH_REPORT.parent.mkdir(parents=True, exist_ok=True)
    PATH_REPORT.write_text(
        json.dumps({**result, "report": report_dict}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    try:
        import mlflow

        if mlflow.active_run() is not None:
            mlflow.log_metrics({"accuracy": accuracy, "f1_weighted": f1})
    except ModuleNotFoundError:
        pass

    if accuracy < threshold:
        raise ValueError(
            f"Accuracy {accuracy:.4f} abaixo do threshold {threshold}. "
            "Pipeline interrompida — revise o modelo ou os dados."
        )

    logger.info("Avaliacao aprovada.")
    return result


if __name__ == "__main__":
    run_evaluation()
