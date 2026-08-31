import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def load_data(raw_path: str | Path) -> pd.DataFrame:
    """
    Carrega um CSV do dataset Medical Abstracts TC Corpus.

    Retorna um DataFrame.
    """
    logger.info("Importando dados - Medical Abstracts TC Corpus: %s", raw_path)
    df = pd.read_csv(raw_path)
    logger.info(
        "Dataset carregado: %d linhas x %d colunas | colunas=%s",
        df.shape[0],
        df.shape[1],
        list(df.columns),
    )
    return df
