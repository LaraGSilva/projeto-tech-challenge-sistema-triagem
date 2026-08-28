# 🩺 Medical Abstracts TC Corpus – Model Canvas

## 📖 Descrição
O dataset **Medical Abstracts TC Corpus** contém abstracts médicos com a descrição de doenças e suas respectivas categorias.  
O objetivo do modelo é **classificar a doença** com base no texto do abstract.

---

## 🔎 Análise Exploratória
Foi realizada uma análise para entender a **distribuição das classes** e identificar possíveis desbalanceamentos.

### 📊 Distribuição das Condições

| Label | Categoria                      | Quantidade |
|-------|--------------------------------|------------|
| 1     | Neoplasms                      | 3163       |
| 2     | Digestive system diseases      | 1494       |
| 3     | Nervous system diseases        | 1925       |
| 4     | Cardiovascular diseases        | 3051       |
| 5     | General pathological conditions| 4805       |

## ⚖️ Observação
Há **diferença significativa na volumetria** entre as classes, tornando necessário aplicar **técnicas de balanceamento** (class weights, oversampling ou SMOTE) para garantir melhor desempenho do modelo.

![alt text](image.png)
