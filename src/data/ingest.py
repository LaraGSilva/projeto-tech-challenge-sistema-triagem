import logging
import os
from pathlib import Path
import pandas as pd
# Configuração básica do logging

logging.basicConfig(
    level=logging.INFO,  # define o nível mínimo de log
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)



def load_data(raw_path) -> pd.DataFrame:
    """
    Carrega o arquivo csv com os dados de treino do dataset medical abstract tc corpus

    Retorna um dataframe
    """

    logger.info("importando dados - Medical Abstracts TC Corpus dataset")
    df = pd.read_csv(raw_path)
    logger.info(f"Dataset carregado com sucesso: {df.shape[0]} linhas x {df.shape[1]} colunas")
    logger.info(f"Colunas presentes no dataset = {(df.columns)}")
    return df