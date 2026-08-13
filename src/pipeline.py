"""Pipeline end-to-end de ML.

Orquestra: ingest -> validate -> train -> evaluate -> save.

Execução:
    cd src
    python pipeline.py
"""
import logging
from pathlib import Path

import joblib

from data.ingest import load_data
from data.validate import validate_data
from train.train import train_model
from train.evaluate import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("pipeline")


def run_pipeline():
    """Executa pipeline completo."""
    logger.info("🚀 Iniciando pipeline...")

    # 1. Ingest

    # 2. Validate

    # 3. Train

    # 4. Evaluate

    # 5. Deploy (salvar localmente)

    # 6. Metricas


if __name__ == "__main__":
    run_pipeline()