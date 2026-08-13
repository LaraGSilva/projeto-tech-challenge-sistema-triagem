import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sys
import os # Importar a biblioteca OS

def main():
    print("Iniciando Dummy Training...")
    try:
        df = pd.read_csv('dataset_processado.csv')
        X = df[['feature1', 'feature2']]
        y = df['target']
        
        clf = DecisionTreeClassifier(max_depth=2)
        clf.fit(X, y)
        
        # Simulação de salvamento do modelo em disco (artefato binário)
        import pickle
        with open('modelo.pkl', 'wb') as f:
            pickle.dump(clf, f)
            
        # ==========================================
        # GOVERNANÇA: Integração com Model Registry
        # ==========================================
        commit_sha = os.environ.get("GITHUB_SHA", "Desconhecido")
        
        print("\n--- Integrando com o Model Registry (Ex: MLflow) ---")
        print("Enviando o arquivo modelo.pkl para o armazenamento central...")
        print(f"ETIQUETA DE RASTREABILIDADE: Modelo amarrado ao Commit Git: {commit_sha}")
        print("SUCESSO: O cartório de modelos aceitou o registro.")
        # ==========================================
        
        sys.exit(0)
        
    except Exception as e:
        print(f"ERRO DE COMPUTAÇÃO: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()