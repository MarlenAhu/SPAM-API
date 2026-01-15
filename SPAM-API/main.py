from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Spam Detector API")

# Cargar modelo
modelo_api = joblib.load('modelo_spam_svm.pkl')

class Mensaje(BaseModel):
    texto: str

@app.get("/")
def inicio():
    return {"mensaje": "API funcionando", "status": "OK"}

@app.post("/predecir")
def predecir(msg: Mensaje):
    pred = modelo_api.predict([msg.texto])[0]
    decision = modelo_api.decision_function([msg.texto])[0]
    return {
        "texto": msg.texto,
        "prediccion": int(pred),
        "confianza": float(abs(decision))
    }