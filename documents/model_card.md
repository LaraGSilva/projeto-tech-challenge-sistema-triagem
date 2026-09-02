# Model Card — Classificador de Categoria de Doença

## Visão geral

| | |
|---|---|
| **Tarefa** | Classificação de texto (NLP), multiclasse — 5 categorias |
| **Entrada** | Texto livre do laudo / abstract médico (inglês) |
| **Saída** | `condition_label` (1–5), `condition_name`, `confianca` (prob. máx.), vetor de `probabilidades` por classe |
| **Tipo** | `sklearn.Pipeline` = `TfidfVectorizer` → `LogisticRegression` |
| **Artefatos** | `models/classifier.pkl` · `classifier.onnx` · `classifier.int8.onnx` (ONNX quantizado, usado em produção) |
| **Metadados** | `models/metadata.json` · **Métricas** `models/evaluation.json` · **Otimização** `models/optimization.json` |
| **Versão** | 1.0.0 |

## Uso pretendido

- **Uso previsto:** triagem automática — direcionar laudos para a especialidade
  provável no momento da emissão (arquitetura *online*, servida via API REST).
- **Fora de escopo:** diagnóstico clínico, decisão sobre urgência de paciente,
  textos que não sejam laudos/abstracts, idiomas diferentes de inglês. A saída é
  um apoio de roteamento, **não** um parecer médico.

## Classes

| label | condição | treino | teste (hold-out) |
|---|---|---|---|
| 1 | neoplasms | 2.024 | 506 |
| 2 | digestive system diseases | 956 | 239 |
| 3 | nervous system diseases | 1.232 | 308 |
| 4 | cardiovascular diseases | 1.953 | 488 |
| 5 | general pathological conditions | 3.075 | 769 |

## Dados de treino

- **Fonte:** *Medical Abstracts TC Corpus* (`src/data/raw/medical_tc_train.csv`,
  11.550 laudos). Versionado com DVC.
- **Split:** `train_test_split` estratificado, 80/20, `random_state=42` →
  **9.240 treino / 2.310 teste**. (O arquivo `medical_tc_test.csv`, 2.888 linhas,
  não é usado no fluxo atual — a avaliação usa o hold-out interno.)
- Análise exploratória e decisões: [analise_dados.md](./analise_dados.md).

## Pré-processamento e features

1. `clean_text` (`src/model/text.py`): minúsculas, remove conteúdo entre
   colchetes, remove dígitos e pontuação, colapsa espaços. Leve de propósito
   (só `re`) — o mesmo código roda no treino e na inferência.
2. `TfidfVectorizer`: `stop_words="english"`, `ngram_range=(1, 2)`, `min_df=5`,
   `max_df=0.9`, `sublinear_tf=True` → **vocabulário de 27.831 termos**.
3. `LogisticRegression`: `C=0.3`, `max_iter=2000`, `class_weight="balanced"`
   (compensa o desbalanceamento sem oversampling).

## Métricas de avaliação (hold-out, 2.310 laudos)

| | valor |
|---|---|
| Accuracy | **0,620** |
| F1 (weighted) | **0,605** |
| F1 (macro) | 0,622 |
| Accuracy de treino | 0,706 |

### Por classe

| classe | precision | recall | f1-score | suporte |
|---|---|---|---|---|
| 1 neoplasms | 0,72 | 0,76 | 0,74 | 506 |
| 2 digestive system diseases | 0,51 | 0,77 | 0,61 | 239 |
| 3 nervous system diseases | 0,50 | 0,71 | 0,59 | 308 |
| 4 cardiovascular diseases | 0,67 | 0,78 | 0,72 | 488 |
| 5 general pathological conditions | 0,64 | 0,34 | 0,44 | 769 |

*Gate de qualidade:* `run_evaluation` lança erro se a accuracy ficar abaixo de
**0,55** — a DAG do Airflow falha e o modelo não é promovido.

## Limitações e vieses

- **Classe 5 (general pathological conditions):** recall baixo (0,34) — é uma
  categoria "guarda-chuva" e o modelo tende a distribuí-la entre as outras.
- **Desbalanceamento:** classe 5 tem ~3× a classe 2; mitigado com
  `class_weight="balanced"`, não eliminado.
- **Domínio:** treinado em abstracts em inglês; laudos em português ou com muita
  abreviação/ruído devem degradar.
- **Confiança calibrada?** Não há calibração de probabilidade — `confianca` é a
  probabilidade bruta do `LogisticRegression`.
- **Latência:** inferência pura ~1,1 ms (sklearn). Com o backend ONNX int8
  (`MODEL_BACKEND=onnx-int8`, padrão em produção) cai para ~0,25 ms sem perda de
  acurácia — ver [comparacao.md](./comparacao.md).

## Retreino

`python -m src.pipeline` (local) ou a DAG `ml_medical_pipeline` no Airflow
(`ingest → validate → preprocess → train → evaluate`). Parâmetros em
`config/config.yaml`. Cada execução regrava `models/classifier.pkl`,
`models/metadata.json` e `models/evaluation.json`.

Após retreinar, regenerar os artefatos ONNX:
`python -m src.model.optimize` (converte + quantiza + valida a paridade).
