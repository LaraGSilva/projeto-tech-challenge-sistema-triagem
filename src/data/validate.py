import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Colunas mínimas esperadas no CSV bruto de treino/teste.
REQUIRED_COLUMNS = {
    "condition_label": "integer",
    "medical_abstract": "text",
}
MIN_ROWS = 2000  # requisito do desafio: >= 2.000 amostras


def _as_dataframe(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    return pd.read_csv(data)


def validate_schema(data) -> bool:
    """
    Valida o schema do dataset de laudos médicos.

    Aceita um caminho de CSV ou um DataFrame já carregado. Levanta AssertionError
    quando o contrato é violado (o Airflow marca a task como FAILED).
    """
    logger.info("Validando schema do dataset...")
    df = _as_dataframe(data)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    assert not missing, f"Colunas faltando: {missing}"

    assert pd.api.types.is_integer_dtype(df["condition_label"]), (
        f"'condition_label' deveria ser inteiro, obtido {df['condition_label'].dtype}"
    )
    assert df["medical_abstract"].map(lambda v: isinstance(v, str)).all(), (
        "'medical_abstract' contém valores não textuais"
    )

    assert df["medical_abstract"].str.strip().str.len().gt(0).all(), (
        "Existem laudos vazios em 'medical_abstract'"
    )
    assert len(df) >= MIN_ROWS, (
        f"Dataset com {len(df)} linhas — mínimo exigido é {MIN_ROWS}"
    )

    n_classes = df["condition_label"].nunique()
    assert n_classes >= 2, f"Esperado >= 2 classes, obtido {n_classes}"

    logger.info(
        "Schema valido: %d linhas x %d colunas | %d classes",
        len(df),
        len(df.columns),
        n_classes,
    )
    return True
