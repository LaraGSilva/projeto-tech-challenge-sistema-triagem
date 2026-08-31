"""Testes das funções de dados/pré-processamento (puras, sem arquivos)."""

import pandas as pd
import pytest

from src.data.validate import validate_schema
from src.model.preprocessing import PATH_LABELS, merge_labels
from src.model.text import clean_text


# ── clean_text ───────────────────────────────────────────────────────────────
def test_clean_text_normaliza():
    bruto = "Acute MYOCARDIAL infarction [see Fig. 2], troponin = 3.5 ng/mL!!!"
    assert clean_text(bruto) == "acute myocardial infarction troponin ng ml"


def test_clean_text_entrada_invalida():
    assert clean_text(None) == ""
    assert clean_text(123) == ""


# ── validate_schema ──────────────────────────────────────────────────────────
def _df_valido(n=2000):
    return pd.DataFrame(
        {
            "condition_label": [1, 2, 3, 4, 5] * (n // 5),
            "medical_abstract": ["texto de laudo médico"] * n,
        }
    )


def test_validate_schema_ok():
    assert validate_schema(_df_valido()) is True


def test_validate_schema_coluna_faltando():
    df = _df_valido().drop(columns=["medical_abstract"])
    with pytest.raises(AssertionError, match="Colunas faltando"):
        validate_schema(df)


def test_validate_schema_poucas_linhas():
    with pytest.raises(AssertionError, match="mínimo"):
        validate_schema(_df_valido(n=100))


def test_validate_schema_label_nao_inteiro():
    df = _df_valido()
    df["condition_label"] = df["condition_label"].astype(float)
    with pytest.raises(AssertionError, match="inteiro"):
        validate_schema(df)


# ── merge_labels ─────────────────────────────────────────────────────────────
@pytest.mark.skipif(
    not PATH_LABELS.exists(),
    reason="medical_tc_labels.csv ausente (rastreado por DVC) — rode `dvc pull`",
)
def test_merge_labels_adiciona_nome_da_condicao():
    df = pd.DataFrame({"condition_label": [1, 4], "medical_abstract": ["a", "b"]})
    out = merge_labels(df)
    assert "condition_name" in out.columns
    assert out.loc[out.condition_label == 1, "condition_name"].iloc[0] == "neoplasms"
