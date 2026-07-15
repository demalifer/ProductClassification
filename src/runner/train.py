from partd import encode
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import Adam
from configuration.config import *
import torch
from process.dataset import DataLoader

# 训练器类
class Trainer:
    # 初始化
    def __init__(self, model, train_loader, optimizer, device):
        pass

    # 核心方法
    def train(self):
        pass

if __name__ == '__main__':
    #1. 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #2. 分词器
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    #3. 加载labels
    with open(MODEL_DIR / LABELS_FILE, 'r', encoding='utf-8') as f:
        all_labels = f.read().splitlines()
    id2label = {index: label for index, label in enumerate(all_labels)}
    label2id = {label: index for index, label in enumerate(all_labels)}
    #4. 加载预训练模型
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id
    )
    model.save_pretrained(MODEL_DIR)
    #5.优化器
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    #6.数据加载器
    train_loader = DataLoader(tokenizer)
    #7.训练器
    trainer = Trainer(model, train_loader, optimizer, device)
    #8.训练
    trainer.train()