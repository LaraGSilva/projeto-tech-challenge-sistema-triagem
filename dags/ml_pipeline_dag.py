"""
DAG de treino/retreino do classificador NLP de laudos médicos.

Fluxo: ingest -> validate -> preprocess -> train -> evaluate
(carregamento de dados -> treino -> salvamento do modelo -> gate de qualidade).

Cada task chama diretamente as funções de ``src/`` e usa XCom só para caminhos
de arquivo (não DataFrames). O MLflow é usado como tracking, aninhando um run
por task sob um run pai da execução da DAG.
"""

from datetime import datetime, timedelta

import mlflow
from airflow import DAG
from airflow.operators.python import PythonOperator

from src.data.ingest import load_data
from src.data.validate import validate_schema
from src.model.evaluate import run_evaluation
from src.model.preprocessing import PATH_LABELS, PATH_TRAIN, build_dataset
from src.model.train import run_training

# Caminhos derivados do próprio pacote src/ — funcionam independentemente de
# onde a pasta dags/ está montada no container.
PATH_TRAIN_CSV = str(PATH_TRAIN)
PATH_LABEL_CSV = str(PATH_LABELS)
PATH_PROCESSED_CSV = str(
    PATH_TRAIN.parent.parent / "processed" / "data_processed_final.csv"
)

MLFLOW_EXPERIMENT = "medical_abstracts_classification"

default_args = {
    "owner": "ml_team",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
    "email_on_failure": False,
    "email_on_retry": False,
}


def task_ingest(**context) -> str:
    """Carrega o CSV bruto e registra o volume de dados."""
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="1_ingest", nested=True):
        mlflow.log_param("raw_path", PATH_TRAIN_CSV)
        df = load_data(PATH_TRAIN_CSV)
        mlflow.log_metric("n_rows_raw", len(df))
        mlflow.log_metric("n_cols_raw", len(df.columns))
    context["ti"].xcom_push(key="raw_path", value=PATH_TRAIN_CSV)
    return PATH_TRAIN_CSV


def task_validate(**context) -> bool:
    """Valida o schema do CSV bruto (falha a task se o contrato quebrar)."""
    raw_path = context["ti"].xcom_pull(task_ids="ingest", key="raw_path")
    with mlflow.start_run(run_name="2_validate", nested=True):
        is_valid = validate_schema(raw_path)
        mlflow.log_param("schema_valid", is_valid)
    return is_valid


def task_preprocess(**context) -> str:
    """Limpa o texto, junta os labels e salva o dataset processado."""
    with mlflow.start_run(run_name="3_preprocess", nested=True):
        df = build_dataset(
            raw_path=PATH_TRAIN_CSV,
            labels_path=PATH_LABEL_CSV,
            save_to=PATH_PROCESSED_CSV,
        )
        mlflow.log_metric("n_rows_processed", len(df))
        mlflow.log_param("processed_path", PATH_PROCESSED_CSV)
    context["ti"].xcom_push(key="processed_path", value=PATH_PROCESSED_CSV)
    return PATH_PROCESSED_CSV


def task_train(**context) -> str:
    """Treina o pipeline TF-IDF + LogisticRegression e salva o artefato."""
    processed_path = context["ti"].xcom_pull(task_ids="preprocess", key="processed_path")
    with mlflow.start_run(run_name="4_train", nested=True):
        model_path = run_training(processed_path)
        mlflow.log_param("model_path", model_path)
    context["ti"].xcom_push(key="model_path", value=model_path)
    return model_path


def task_evaluate(**context) -> dict:
    """Avalia o modelo; falha se a accuracy ficar abaixo do threshold."""
    with mlflow.start_run(run_name="5_evaluate", nested=True):
        metrics = run_evaluation()
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    context["ti"].xcom_push(key="metrics", value=metrics)
    return metrics


with DAG(
    dag_id="ml_medical_pipeline",
    description="Treino/retreino do classificador NLP de laudos médicos com tracking MLflow",
    default_args=default_args,
    start_date=datetime(2026, 8, 18),
    schedule_interval=None,  # disparo manual; use "@weekly" para agendar o retreino
    catchup=False,
    tags=["ml", "nlp", "mlflow", "medical"],
    doc_md=__doc__,
) as dag:

    ingest = PythonOperator(task_id="ingest", python_callable=task_ingest)
    validate = PythonOperator(task_id="validate", python_callable=task_validate)
    preprocess = PythonOperator(task_id="preprocess", python_callable=task_preprocess)
    train = PythonOperator(task_id="train", python_callable=task_train)
    evaluate = PythonOperator(task_id="evaluate", python_callable=task_evaluate)

    ingest >> validate >> preprocess >> train >> evaluate
