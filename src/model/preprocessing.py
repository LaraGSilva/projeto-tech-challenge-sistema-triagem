"""
Pré-processamento textual dos laudos médicos.

Mantém-se leve de propósito (regex + normalização), condizente com um
classificador NLP "leve" e com retreino rápido no Airflow. A remoção de
stopwords e a vetorização ficam a cargo do ``TfidfVectorizer``
(``src/model/feature_engineering.py``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.ingest import load_data
from src.model.text import clean_text

__all__ = [
    "clean_text",
    "merge_labels",
    "build_dataset",
    "preprocess",
    "preprocess_merge",
    "TEXT_COL",
    "CLEAN_COL",
    "LABEL_COL",
    "LABEL_NAME_COL",
    "PATH_TRAIN",
    "PATH_TEST",
    "PATH_LABELS",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parents[2]
PATH_TRAIN = _BASE / "src" / "data" / "raw" / "medical_tc_train.csv"
PATH_TEST = _BASE / "src" / "data" / "raw" / "medical_tc_test.csv"
PATH_LABELS = _BASE / "src" / "data" / "raw" / "medical_tc_labels.csv"

TEXT_COL = "medical_abstract"
CLEAN_COL = "medical_abstract_clean"
LABEL_COL = "condition_label"
LABEL_NAME_COL = "condition_name"


def merge_labels(df: pd.DataFrame, labels_path: str | Path = PATH_LABELS) -> pd.DataFrame:
    """Adiciona o nome legível da condição (neoplasms, ...) via ``condition_label``."""
    labels = load_data(labels_path)
    return df.merge(labels, how="left", on=LABEL_COL)


def build_dataset(
    raw_path: str | Path = PATH_TRAIN,
    labels_path: str | Path = PATH_LABELS,
    save_to: str | Path | None = None,
) -> pd.DataFrame:
    """
    Pipeline de dados de ponta a ponta:

    1. carrega o CSV bruto de laudos;
    2. junta o nome da condição (``condition_name``);
    3. cria ``medical_abstract_clean`` pronto para vetorização;
    4. remove linhas cujo texto ficou vazio após a limpeza.

    Opcionalmente persiste o resultado em ``save_to``.
    """
    df = load_data(raw_path)
    df = merge_labels(df, labels_path)

    logger.info("Limpando %d laudos...", len(df))
    df[CLEAN_COL] = df[TEXT_COL].map(clean_text)

    before = len(df)
    df = df[df[CLEAN_COL].str.len() > 0].reset_index(drop=True)
    if before != len(df):
        logger.warning("Removidas %d linhas com texto vazio pós-limpeza", before - len(df))

    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_to, index=False)
        logger.info("Dataset processado salvo em: %s", save_to)

    return df


# ---------------------------------------------------------------------------
# Compatibilidade com chamadas antigas (notebooks / scripts legados).
# ---------------------------------------------------------------------------
def preprocess(text: str) -> str:
    """Alias legado para :func:`clean_text`."""
    return clean_text(text)


def preprocess_merge(
    df: pd.DataFrame, labels_path: str | Path = PATH_LABELS
) -> pd.DataFrame:
    """Junta labels e cria a coluna de texto limpo em um DataFrame já carregado."""
    df = merge_labels(df, labels_path)
    df[CLEAN_COL] = df[TEXT_COL].map(clean_text)
    return df
