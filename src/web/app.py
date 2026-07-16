import torch
import uvicorn
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configuration.config import BERT_MODEL_NAME, MODEL_DIR
from web.schemas import Title, Category
from web.service import TitleService
from runner.predict import Predictor

app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR / 'best')
predictor = Predictor(model=model, tokenizer=tokenizer, device=device)
service = TitleService(predictor=predictor)

@app.get("/predict")
def predict(text: str) -> Category:
    label = service.predict(text)
    return Category(category=label)

def serve():
    uvicorn.run('web.app:app', host="0.0.0.0", port=8000)