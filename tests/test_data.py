import pandas as pd
import os

def test_schema_dados_processados():
    # O teste espera que o preparar_dados.py já tenha rodado no pipeline
    caminho_arquivo = 'dataset_processado.csv'
    
    assert os.path.exists(caminho_arquivo), "O arquivo de dados não foi gerado!"
    
    df = pd.read_csv(caminho_arquivo)
    colunas_esperadas = ['feature1', 'feature2', 'target']

    # 1. Contrato de Colunas
    for col in colunas_esperadas:
        assert col in df.columns, f"QUEBRA DE CONTRATO: A coluna '{col}' sumiu do dataset!"

    # 2. Contrato de Qualidade (Sem nulos)
    assert df.isnull().sum().sum() == 0, "QUALIDADE: Existem valores nulos (NaN) no dataset."

    # 3. Contrato de Regra de Negócio (Target binário)
    valores_target = set(df['target'].unique())
    assert valores_target.issubset({0, 1}), f"ANOMALIA: O target deveria ser 0 ou 1, mas veio: {valores_target}"