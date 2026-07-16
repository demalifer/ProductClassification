import torch
from sklearn.metrics import accuracy_score, f1_score

from runner.train import Trainer, TrainConfig
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from configuration.config import *
from process.dataset import get_dataset

# 验证流程
def evaluate():
    # 1. 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2. 分词器
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    # 3. 加载labels
    with open(MODEL_DIR / LABELS_FILE, "r", encoding="utf-8") as f:
        all_labels = f.read().splitlines()
    # 4. 加载预训练模型
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR / 'best')
    model.save_pretrained(MODEL_DIR)
    # 5.数据集和整理函数
    test_dataset = get_dataset('test')
    collate_fn = DataCollatorWithPadding(
        tokenizer=tokenizer, padding=True, return_tensors="pt"
    )

    # 6. 评估函数
    def compute_metrics(labels, predictions) -> dict:
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")
        return {"acc": acc, "f1": f1}
    
    trainer = Trainer(model, None, test_dataset, collate_fn=collate_fn, compute_metrics=compute_metrics, device=device)
    metrics = trainer.evaluate()
    print(metrics)    

if __name__ == '__main__':
    evaluate()