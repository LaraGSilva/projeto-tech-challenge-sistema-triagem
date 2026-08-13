import logging
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA = {
    "condition_label": np.int64,
    "medical_abstract": str,
    "conditional_name": str
}


def validate_schema(raw_path) -> bool:
    """
    Valida colunas e tipo de dados
    """

    logger.info("validando schema...")

    df = pd.read_csv(raw_path)

    missing = set(SCHEMA) - set(df.columns)
    assert not missing, f"Colunas faltando: {missing}"

    for col, expected_type in SCHEMA.items():
        actual_type = df[col].dtype.type
        assert actual_type == expected_type, (
            f"Tipo errado em {col}, esperado {
                expected_type}, obtido {actual_type}"
        )

    logger.info(f"✅ Schema válido: {len(df)} linhas x {
                len(df.columns)} colunas")
    return True
