import spacy
import re
import pandas as pd
from pathlib import Path
import logging
from src.data import ingest


logging.basicConfig(
    level=logging.INFO,  # define o nível mínimo de log
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


LANGUAGE = "en_core_web_sm"
NLP = spacy.load("en_core_web_sm")
PATH_DATA = Path(__file__).parent.parent / "data" / "raw" / 'medical_tc_train.csv'
PATH_LABEL = Path(__file__).parent.parent / "data" / "raw" / 'medical_tc_labels.csv'


def lower_replace(text: str) -> str:
    """
    Realiza o processamento de formatação + lower case nos dados de entrada textuais.
    
    Retorna as strings formatadas conforme o esperado: lower case + sem acentuações, pontos, virgulas, etc.
    """
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)     
    text = re.sub(r'[^\w\s]', '', text)       
    return text
 
def token_lemma_stop(text: str) -> list:
    """
    Tokenização + lematização + remoção de stopwords
    """
    doc = NLP(text)
    return [token.lemma_ for token in doc if not token.is_stop]

def filter_pos(tokens: list) -> str:
    """
    Filtrar apenas certas classes gramaticais (exemplo: substantivos e adjetivos)
    """
    doc = NLP(" ".join(tokens))
    return " ".join([token.text for token in doc if token.pos_ in ["NOUN", "ADJ", "PRON", "VERB"]])

def preprocess(text: str) -> list:
    """
    Pipeline único

    Realiza todas as etapas de processamento de dados textuais

    Retorna uma lista de valores da coluna string com os dados processados: lower case, sem stop words, etc.
    """
    text = lower_replace(text)
    tokens = token_lemma_stop(text)
    return filter_pos(tokens)

def preprocess_merge(df: pd.DataFrame, str: PATH_LABEL) -> pd.DataFrame:
    """
    Realiza o processo de merge com os labels de cada descrição e condição.

    Retorna o dataframe com o label de cada condition
    """
    df_label = ingest.load_data(PATH_LABEL)
    df_train_merge = df.merge(df_label, how='inner', on='condition_label')
    return df_train_merge


if __name__ == "__main__":

    logger.info(" ETAPA 1- Realizado o a ingestão do dataset original")
    df = ingest.load_data(PATH_DATA)

    # df = df.head(5)

    logger.info("ETAPA 2- Realizando o pre processamento dos dados textuais")
    df['medical_abstract'] = df['medical_abstract'].apply(preprocess)

    logger.info("ETAPA 3- Realizando o merge dos dados para capturar a infromação do descritivo de cada condtition")
    df_final = preprocess_merge(df, PATH_LABEL)

    logger.info("ETAPA 4- Salvando em outro dataframe o dado preprocessado final")
    df_final.to_csv('src/data/processed/data_processed_final.csv')

    print(df_final.head(10))