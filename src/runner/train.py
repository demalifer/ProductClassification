from attr import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from torch.optim import Adam
from configuration.config import *
import torch
from process.dataset import get_dataset
from torch.utils.data import DataLoader

# 训练配置类
@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-5
    save_steps: int = 100
    output_dir: str = './models'
    log_dir: str = './logs'

# 训练器类
class Trainer:
    # 初始化
    def __init__(self, model, train_dataset, collate_fn, device, train_config=None):
        # 训练参数
        self.train_config = train_config
        # 模型和设备
        self.model = model.to(device)
        self.device = device
        # 数据集和数据整理函数
        self.train_dataset = train_dataset
        self.collate_fn = collate_fn
        # 优化器
        self.optimizer = Adam(model.parameters(), lr=self.train_config.learning_rate)
    
    #获取数据加载器
    def _get_dataloader(self):
        # 设置格式为tensor
        self.train_dataset.set_format(type="torch")

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        return dataloader

    # 核心方法
    def train(self):


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
    #5.数据集和整理函数
    train_dataset = get_dataset('train')
    collate_fn = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors='pt'
    )
    #6. 定义训练配置
    train_config = TrainConfig(batch_size=32, output_dir=MODEL_DIR, log_dir=MODEL_DIR, save_steps=100)
    #7. 训练器
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        device=device,
        train_config=train_config
    )
    #8. 训练
    trainer.train()