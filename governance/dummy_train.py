import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sys

def main():
    print("Iniciando Dummy Training (Treinamento de Sanidade)...")
    
    try:
        # Carrega os dados passados pelo artefato
        df = pd.read_csv('dataset_processado.csv')
        
        # Separa features e target
        X = df[['feature1', 'feature2']]
        y = df['target']
        
        # Treina um modelo super rápido
        clf = DecisionTreeClassifier(max_depth=2)
        clf.fit(X, y)
        
        score = clf.score(X, y)
        print(f"SUCESSO: Modelo treinado sem erros de sintaxe! Acurácia no micro-dataset: {score}")
        sys.exit(0)
        
    except Exception as e:
        print(f"ERRO DE COMPUTAÇÃO: O código do modelo falhou. Detalhes: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()