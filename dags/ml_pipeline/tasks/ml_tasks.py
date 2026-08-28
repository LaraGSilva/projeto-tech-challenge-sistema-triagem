from datetime import datetime, timedelta
from pathlib import Path

import mlflow
from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Imports ──────────────────────────────────────────────────────────────────
from src.data.ingest import load_data
from src.data.validate import validate_schema
from src.model.preprocessing import preprocess_merge
from src.model.feature_engineering import transform_term_frequency
from src.model.train import run_training    
from src.model.evaluate import run_evaluation

# ── Caminhos dos dados brutos ─────────────────────────────────────────────────
_BASE = Path(__file__).parent.parent
PATH_TRAIN_CSV = str(_BASE / "data" / "raw" / "medical_tc_train.csv")
PATH_LABEL_CSV = str(_BASE / "data" / "raw" / "medical_tc_labels.csv")

MLFLOW_EXPERIMENT = "medical_abstracts_classification"

# ── Configurações padrão da DAG ───────────────────────────────────────────────
default_args = {
    "owner": "ml_team",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
    "email_on_failure": False,
    "email_on_retry": False,
}


# ═══════════════════════════════════════════════════════════════════════════════
# WRAPPERS DAS TASKS
# Cada função chama diretamente sua função do módulo original e
# usa XCom para passar resultados entre tasks.
# ═══════════════════════════════════════════════════════════════════════════════

def task_ingest(**context) -> str:
    """
    Carrega os dados brutos via ingest.load_data().
    Empurra o path do CSV via XCom para as tasks seguintes.
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="1_ingest", nested=True):
        mlflow.log_param("raw_path", PATH_TRAIN_CSV)

        # Chama sua função original — sem modificação
        df = load_data(PATH_TRAIN_CSV)

        mlflow.log_metric("n_rows_raw", len(df))
        mlflow.log_metric("n_cols_raw", len(df.columns))

    # Passa o path (não o DataFrame) via XCom — DataFrames podem ser grandes
    context["ti"].xcom_push(key="raw_path", value=PATH_TRAIN_CSV)
    return PATH_TRAIN_CSV


def task_validate(**context) -> bool:
    """
    Valida o schema do CSV via validate.validate_schema().
    Levanta AssertionError se o schema estiver errado (Airflow marca a task como FAILED).
    """
    raw_path = context["ti"].xcom_pull(task_ids="ingest", key="raw_path")

    with mlflow.start_run(run_name="2_validate", nested=True):
        # Chama sua função original — sem modificação
        is_valid = validate_schema(raw_path)
        mlflow.log_param("schema_valid", is_valid)

    context["ti"].xcom_push(key="raw_path", value=raw_path)
    return is_valid


def task_preprocessing(**context) -> str:
    """
    Aplica pré-processamento textual e merge com labels
    via preprocessing.preprocess_merge().
    """
    raw_path = context["ti"].xcom_pull(task_ids="validate", key="raw_path")

    with mlflow.start_run(run_name="3_preprocessing", nested=True):
        df = load_data(raw_path)

        # Chama sua função original — sem modificação
        df_processed = preprocess_merge(df, PATH_LABEL_CSV)

        processed_path = str(
            _BASE / "data" / "processed" / "data_processed_final.csv"
        )
        mlflow.log_metric("n_rows_processed", len(df_processed))
        mlflow.log_param("processed_path", processed_path)

    context["ti"].xcom_push(key="processed_path", value=processed_path)
    return processed_path


def task_feature_engineering(**context) -> str:
    """
    Gera features TF-IDF via feature_engineering.transform_term_frequency().
    """
    processed_path = context["ti"].xcom_pull(
        task_ids="preprocessing", key="processed_path"
    )

    with mlflow.start_run(run_name="4_feature_engineering", nested=True):
        df = load_data(processed_path)

        # Chama sua função original — sem modificação
        tfidf_df = transform_term_frequency(df)

        features_path = str(
            _BASE / "data" / "features" / "tfidf_features.csv"
        )
        mlflow.log_metric("n_features", tfidf_df.shape[1] - 1)  # -1 pelo label
        mlflow.log_param("features_path", features_path)

    context["ti"].xcom_push(key="features_path", value=features_path)
    return features_path


def task_train(**context) -> str:
    """
    Treina o modelo via train.run_training().
    """
    features_path = context["ti"].xcom_pull(
        task_ids="feature_engineering", key="features_path"
    )

    with mlflow.start_run(run_name="5_train", nested=True):
        # Chama sua função original — sem modificação
        model_path = run_training(features_path)
        mlflow.log_param("model_path", model_path)

    context["ti"].xcom_push(key="model_path", value=model_path)
    return model_path


def task_evaluate(**context) -> dict:
    """
    Avalia o modelo via evaluate.run_evaluation().
    Falha automaticamente se accuracy < threshold definido em evaluate.py.
    """
    model_path = context["ti"].xcom_pull(task_ids="train", key="model_path")
    test_path = str(_BASE / "data" / "features" / "test_set.pkl")

    with mlflow.start_run(run_name="6_evaluate", nested=True):
        # Chama sua função original — sem modificação
        metrics = run_evaluation(model_path, test_path)

    context["ti"].xcom_push(key="metrics", value=metrics)
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# DEFINIÇÃO DA DAG
# ═══════════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id="ml_medical_pipeline",
    description="Pipeline de classificação de abstracts médicos com MLflow tracking",
    default_args=default_args,
    start_date=datetime(2026, 8, 18),
    schedule_interval=None,   # disparo manual; mude para "@weekly" se quiser agendamento
    catchup=False,
    tags=["ml", "nlp", "mlflow", "medical"],
    doc_md=__doc__,
) as dag:

    ingest = PythonOperator(
        task_id="ingest",
        python_callable=task_ingest,
        doc_md="Carrega CSV bruto via ingest.load_data()",
    )

    validate = PythonOperator(
        task_id="validate",
        python_callable=task_validate,
        doc_md="Valida schema via validate.validate_schema()",
    )

    preprocessing = PythonOperator(
        task_id="preprocessing",
        python_callable=task_preprocessing,
        doc_md="Pré-processa texto e faz merge com labels via preprocessing.preprocess_merge()",
    )

    feature_engineering = PythonOperator(
        task_id="feature_engineering",
        python_callable=task_feature_engineering,
        doc_md="Gera features TF-IDF via feature_engineering.transform_term_frequency()",
    )

    train = PythonOperator(
        task_id="train",
        python_callable=task_train,
        doc_md="Treina classificador via train.run_training()",
    )

    evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=task_evaluate,
        doc_md="Avalia modelo via evaluate.run_evaluation()",
    )

    # ── Ordem da pipeline ─────────────────────────────────────────────────────
    ingest >> validate >> preprocessing >> feature_engineering >> train >> evaluate