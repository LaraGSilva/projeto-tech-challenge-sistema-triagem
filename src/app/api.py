from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Anti-Fraud ML API")

# O Contrato de Entrada (Payload)
class Transacao(BaseModel):
    feature1: float
    feature2: float

@app.post("/predict")
def predict(dados: Transacao):
    # Em um cenário real, carregaríamos o modelo.pkl do MLflow aqui.
    # Para o teste do pipeline, vamos simular a inferência.
    score = dados.feature1 + dados.feature2
    aprovado = int(score > 5.0)
    
    return {"aprovado": aprovado, "score": score}