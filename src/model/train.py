"""
Treino do classificador NLP de laudos médicos.

Modelo: ``Pipeline([TfidfVectorizer, LogisticRegression])`` — leve, rápido de
treinar/retreinar e exportável para ONNX (Etapa 4). O artefato salvo já embute
a vetorização, então a API só precisa de ``model.predict([texto])``.

Alvo: ``condition_label`` (categoria da doença) a partir do texto do laudo.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.model.feature_engineering import build_vectorizer
from src.model.preprocessing import (
    CLEAN_COL,
    LABEL_COL,
    LABEL_NAME_COL,
    TEXT_COL,
    build_dataset,
    clean_text,
)

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - yaml quase sempre presente
    yaml = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parents[2]
PATH_CONFIG = _BASE / "config" / "config.yaml"
PATH_MODEL = _BASE / "models" / "classifier.pkl"
PATH_TEST_SPLIT = _BASE / "models" / "test_split.pkl"
PATH_METADATA = _BASE / "models" / "metadata.json"

DEFAULT_PARAMS: dict = {
    "test_size": 0.2,
    "random_state": 42,
    "classifier": {"C": 0.3, "max_iter": 2000, "class_weight": "balanced"},
}


def load_config(path: str | Path = PATH_CONFIG) -> dict:
    """Lê ``config/config.yaml`` (seção ``training``) com fallback nos padrões."""
    if yaml is None or not Path(path).exists():
        logger.warning("config.yaml indisponível — usando parâmetros padrão")
        return {**DEFAULT_PARAMS, "classifier": dict(DEFAULT_PARAMS["classifier"])}

    with open(path, "r", encoding="utf-8") as fh:
        cfg = (yaml.safe_load(fh) or {}).get("training", {})

    merged = {**DEFAULT_PARAMS, **cfg}
    merged["classifier"] = {
        **DEFAULT_PARAMS["classifier"],
        **cfg.get("classifier", {}),
    }
    return merged


def build_pipeline(classifier_params: dict) -> Pipeline:
    """Monta o pipeline TF-IDF -> Logistic Regression."""
    return Pipeline(
        steps=[
            ("tfidf", build_vectorizer()),
            ("clf", LogisticRegression(**classifier_params)),
        ]
    )


def _maybe_mlflow(kind: str, payload) -> None:
    """Loga no MLflow apenas se houver um run ativo (não quebra execução avulsa)."""
    try:
        import mlflow

        if mlflow.active_run() is None:
            return
        if kind == "params":
            mlflow.log_params(payload)
        elif kind == "metrics":
            mlflow.log_metrics(payload)
        elif kind == "model":
            import mlflow.sklearn

            mlflow.sklearn.log_model(payload, artifact_path="model")
    except ModuleNotFoundError:
        pass
    except Exception as exc:  # pragma: no cover
        logger.warning("MLflow log ignorado: %s", exc)


def _ensure_clean_column(df: pd.DataFrame) -> pd.DataFrame:
    if CLEAN_COL not in df.columns:
        logger.info("Coluna '%s' ausente — aplicando clean_text()", CLEAN_COL)
        df = df.copy()
        df[CLEAN_COL] = df[TEXT_COL].map(clean_text)
    return df


def run_training(
    dataset: pd.DataFrame | str | Path | None = None,
    config_path: str | Path = PATH_CONFIG,
) -> str:
    """
    Treina o classificador e persiste o artefato.

    ``dataset`` pode ser:
      * ``None``  -> constrói o dataset do zero via ``build_dataset()``;
      * caminho   -> CSV já processado (com ``medical_abstract_clean``);
      * DataFrame -> usado diretamente.

    Retorna o caminho do modelo salvo.
    """
    cfg = load_config(config_path)

    if dataset is None:
        df = build_dataset()
    elif isinstance(dataset, (str, Path)):
        logger.info("Carregando dataset processado de: %s", dataset)
        df = pd.read_csv(dataset)
    else:
        df = dataset

    df = _ensure_clean_column(df)
    df = df[df[CLEAN_COL].str.len() > 0].reset_index(drop=True)

    X = df[CLEAN_COL]
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=y,
    )

    clf_params = cfg["classifier"]
    logger.info(
        "Treinando TF-IDF + LogisticRegression | %d amostras treino | params=%s",
        len(X_train),
        clf_params,
    )
    pipeline = build_pipeline(clf_params)
    pipeline.fit(X_train, y_train)

    train_accuracy = float(pipeline.score(X_train, y_train))
    logger.info("Accuracy (treino): %.4f", train_accuracy)

    # Mapa label -> nome legível, consumido pela API.
    label_names: dict[int, str] = {}
    if LABEL_NAME_COL in df.columns:
        label_names = {
            int(k): str(v)
            for k, v in df.drop_duplicates(LABEL_COL)
            .set_index(LABEL_COL)[LABEL_NAME_COL]
            .to_dict()
            .items()
        }

    PATH_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, PATH_MODEL)
    joblib.dump({"X_test": X_test, "y_test": y_test}, PATH_TEST_SPLIT)

    metadata = {
        "task": "medical_abstract_condition_classification",
        "classes": [int(c) for c in pipeline.classes_],
        "label_names": label_names,
        "classifier_params": clf_params,
        "tfidf_vocab_size": len(pipeline.named_steps["tfidf"].vocabulary_),
        "train_accuracy": train_accuracy,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    PATH_METADATA.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _maybe_mlflow("params", clf_params)
    _maybe_mlflow("metrics", {"train_accuracy": train_accuracy})
    _maybe_mlflow("model", pipeline)

    logger.info("Modelo salvo em: %s", PATH_MODEL)
    return str(PATH_MODEL)


if __name__ == "__main__":
    run_training()
