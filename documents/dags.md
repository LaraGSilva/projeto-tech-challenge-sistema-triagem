# Orquestração — Airflow

## Objetivo

Automatizar o ciclo de **treino / retreino** do modelo: carregar os dados,
validar o schema, pré-processar o texto, treinar e avaliar — com rastreio de
parâmetros e métricas no MLflow e um *gate* de qualidade que interrompe a pipeline
se o modelo regredir.

## Stack (`docker-compose.yaml`)

Baseado no template oficial do Airflow, com **CeleryExecutor**:

| Serviço | Papel |
|---|---|
| `airflow-webserver` | UI em http://localhost:8080 (`airflow` / `airflow`) |
| `airflow-scheduler` | agenda e dispara as DAGs |
| `airflow-worker` | executa as tasks (Celery) |
| `airflow-triggerer` | deferrable operators |
| `postgres` | metadata DB do Airflow |
| `redis` | broker do Celery |
| `mlflow` | tracking server em http://localhost:5000 (`MLFLOW_TRACKING_URI` já aponta pra cá) |

A imagem é o `Dockerfile.airflow` (Airflow oficial + `requirements-airflow.txt`:
`pandas`, `scikit-learn`, `joblib`, `mlflow-skinny`). O projeto inteiro é montado
em `/opt/airflow/project` e adicionado ao `PYTHONPATH`, então as tasks importam
`src.*` diretamente.

```bash
docker compose up airflow-init      # 1ª vez: cria o metadata DB + usuário admin
docker compose up -d                # sobe a stack
```

> Requer Docker com ≥ 4 GB de RAM.

## DAG `ml_medical_pipeline` (`dags/ml_pipeline_dag.py`)

- `schedule_interval=None` — disparo manual (trocar para `@weekly` para agendar o
  retreino).
- `catchup=False`, `retries=1`, `retry_delay=3min`.
- Cada task chama **diretamente** a função de `src/` correspondente e passa
  **caminhos de arquivo** (não DataFrames) via XCom.
- Tracking: `mlflow.set_experiment("medical_abstracts_classification")` e um run
  aninhado (`nested=True`) por task.

### Fluxo

```
ingest → validate → preprocess → train → evaluate
```

| Task | Função de `src/` | O que faz |
|---|---|---|
| `ingest` | `src.data.ingest.load_data` | carrega `medical_tc_train.csv`; loga `n_rows_raw` / `n_cols_raw` |
| `validate` | `src.data.validate.validate_schema` | checa colunas, tipos, ≥ 2.000 linhas e ≥ 2 classes; `AssertionError` → task FAILED |
| `preprocess` | `src.model.preprocessing.build_dataset` | junta os nomes das condições, aplica `clean_text` (regex, minúsculas, sem pontuação/dígitos) e salva `data/processed/data_processed_final.csv` |
| `train` | `src.model.train.run_training` | `Pipeline([TfidfVectorizer, LogisticRegression])`, `train_test_split` estratificado, salva `models/classifier.pkl`, `models/test_split.pkl` e `models/metadata.json` |
| `evaluate` | `src.model.evaluate.run_evaluation` | accuracy / F1-weighted / classification report no hold-out; **`ValueError` se accuracy < 0,55** (gate) → falha a DAG e o modelo não é promovido |

### Módulo alternativo

`dags/ml_pipeline/tasks/ml_tasks.py` é uma versão anterior da montagem da DAG
(mantida como referência histórica). A DAG ativa é `dags/ml_pipeline_dag.py`.

## Dados (DVC)

Os CSVs brutos (`src/data/raw/*.csv`) são versionados por **DVC** (ponteiros
`*.dvc` no git). Para rodar a pipeline fora da máquina de origem é preciso um
remote DVC configurado (`dvc pull`). O workflow `.github/workflows/ml-pipeline.yml`
reproduz esse fluxo no CI: `dvc pull` → `build_dataset` → `run_training` +
`run_evaluation` → upload dos artefatos.
