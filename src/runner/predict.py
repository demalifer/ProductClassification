import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configuration.config import *


# 预测器类
class Predictor:
    # 初始化
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        
    # 核心预测方法
    def predict(self, texts: str | list):
        is_str = isinstance(texts, str)
        if is_str:
            texts = [texts]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1).tolist()
        labels = [self.model.config.id2label[pred_id] for pred_id in predictions]
        if is_str:
            return labels[0]
        return labels

# 测试主流程            
def predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR / 'best')
    predictor = Predictor(model, tokenizer, device)
    text = '好奇心钻装纸尿裤'
    result = predictor.predict(text)
    print(result)


if __name__ == '__main__':
    predict()
