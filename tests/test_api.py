from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_predict_endpoint_sucesso():
    # Simulando um cliente chamando nossa API corretamente
    response = client.post("/predict", json={"feature1": 3.0, "feature2": 4.0})
    
    assert response.status_code == 200
    assert response.json()["aprovado"] == 1

def test_predict_endpoint_erro_validacao():
    # Simulando um sistema legado mandando texto em vez de número
    response = client.post("/predict", json={"feature1": "texto_invalido", "feature2": 4.0})
    
    # O FastAPI (Pydantic) deve bloquear e retornar erro 422 (Unprocessable Entity)
    assert response.status_code == 422