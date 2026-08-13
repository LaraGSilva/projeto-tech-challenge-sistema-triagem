import logging
from pathlib import Path
import spacy
import re
import pandas as pd
from src.data import ingest, validate
from src.model import train, evaluate, preprocessing, feature_engineering


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("pipeline")

LANGUAGE = "en_core_web_sm"
NLP = spacy.load("en_core_web_sm")
PATH_DATA = Path(__file__).parent.parent / "data" / "raw" / 'medical_tc_train.csv'
PATH_LABEL = Path(__file__).parent.parent / "data" / "raw" / 'medical_tc_labels.csv'


def run_pipeline():
    """Executa pipeline completo."""
    logger.info("🚀 Iniciando pipeline...")

    # 1. Pre processing
    
    logger.info("PRE PROCESSING --- ETAPA 1- Realizado o a ingestão do dataset original")
    df = ingest.load_data(PATH_DATA)

    logger.info("PRE PROCESSING --- ETAPA 2- Realizando o pre processamento dos dados textuais")
    df['medical_abstract'] = df['medical_abstract'].apply(preprocessing.preprocess)

    logger.info("PRE PROCESSING --- ETAPA 3- Realizando o merge dos dados para capturar a infromação do descritivo de cada condtition")
    df_final = preprocessing.preprocess_merge(df, PATH_LABEL)

    logger.info("PRE PROCESSING --- ETAPA 4- Salvando em outro dataframe o dado preprocessado final")
    df_final.to_csv('src/data/processed/data_processed_final.csv')

    # 2. Validate
    logger.info("VALIDATE - ETAPA 1 - Realizando a validação do schema de dados para o treinamento do modelo")
    try:
        resposta = validate.validate_schema(df_final)
        logger.info(f"Retorno da validação do schema final: {resposta}")

        if resposta == True:
            # 3. BUild and Create Feature ENgineering
            logger.info("FEAT ENGINEERING - ETAPA 1 - Criação de colunas númericas para o treinamento do modelo")
            df = feature_engineering.transform_term_frequency(df_final)
        else:
           logger.info("Etapa de validação não foi concluida com sucesso") 
    except Exception as e:
        logger.info(f'Etapa de validação não foi realizada. {e}')
        

    # 4. Train
    logger.info("TREINAMENTO DO MODELO")
    

    # 5. Evaluate

    # 6. Deploy (salvar localmente)

    # 7. Metricas


if __name__ == "__main__":
    run_pipeline()