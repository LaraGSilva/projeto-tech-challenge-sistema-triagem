"""
Pipeline de treino local (equivalente à DAG do Airflow, mas em um único script).

Fluxo: build_dataset -> run_training -> run_evaluation.

Uso:
    python -m src.pipeline
"""

import logging
from pathlib import Path

from src.data.validate import validate_schema
from src.model.evaluate import run_evaluation
from src.model.preprocessing import LABEL_COL, PATH_TRAIN, build_dataset
from src.model.train import run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("pipeline")

PROCESSED_PATH = Path(__file__).resolve().parent / "data" / "processed" / "data_processed_final.csv"


def run_pipeline() -> dict:
    """Executa o pipeline de treino completo."""
    logger.info("Iniciando pipeline de treino...")

    logger.info("[1/4] Validando schema do dataset bruto")
    validate_schema(PATH_TRAIN)

    logger.info("[2/4] Pré-processando dados")
    df = build_dataset(save_to=PROCESSED_PATH)
    logger.info(
        "Dataset pronto: %d amostras | %d classes",
        len(df),
        df[LABEL_COL].nunique(),
    )

    logger.info("[3/4] Treinando modelo")
    model_path = run_training(df)

    logger.info("[4/4] Avaliando modelo")
    metrics = run_evaluation()

    logger.info("Pipeline concluída: %s -> %s", model_path, metrics)
    return metrics


if __name__ == "__main__":
    run_pipeline()
