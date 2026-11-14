from fastapi import FastAPI
from pydantic import BaseModel
from app.predict import predict_text

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    result = predict_text(input.text)
    return result