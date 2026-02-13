from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import sys
import os

sys.path.append(os.path.abspath("."))

from src.multimodal_model import MultimodalModel

app = FastAPI(title="Drug Response Prediction API")

model = MultimodalModel()
model.load_state_dict(torch.load("model.pt"))
model.eval()

class InputData(BaseModel):
    drug: list
    rna: list

@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: InputData):
    if len(data.drug) != 10:
        raise HTTPException(status_code=400, detail="Drug vector must be length 10")
    if len(data.rna) != 10:
        raise HTTPException(status_code=400, detail="RNA vector must be length 10")

    drug = torch.tensor(data.drug).float().unsqueeze(0)
    rna = torch.tensor(data.rna).float().unsqueeze(0)

    with torch.no_grad():
        pred = model(drug, rna)

    return {"predicted_response": float(pred.item())}
