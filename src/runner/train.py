import torch
import time
from attr import dataclass
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from torch.optim import Adam
from configuration.config import *
from process.dataset import get_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

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
    def __init__(self, model, train_dataset, valid_dataset, collate_fn, compute_metrics, device, train_config=None):
        # 训练参数
        self.train_config = train_config
        # 模型和设备
        self.model = model.to(device)
        self.device = device
        # 数据集和数据整理函数
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.collate_fn = collate_fn
        # 评估函数
        self.compute_metrics = compute_metrics
        # 优化器
        self.optimizer = Adam(model.parameters(), lr=self.train_config.learning_rate)
        # 全局迭代次数
        self.step = 1
        # Tensorboard写入
        self.writer = SummaryWriter(log_dir=str(self.train_config.log_dir / time.strftime("%Y-%m-%d-%H-%M-%S")))
        # 全局最小loss
        self.min_loss = float('inf')

    #获取数据加载器
    def _get_dataloader(self, dataset):
        # 设置格式为tensor
        dataset.set_format(type="torch")

        dataloader = DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        return dataloader

    # 核心方法
    def train(self):
        self.model.train()
        # 获取训练集加载器
        dataloader = self._get_dataloader(self.train_dataset)
        # 训练
        for epoch in range(self.train_config.epochs):
            for inputs in tqdm(dataloader, desc=f'[Epoch {epoch+1}]'):
                this_loss = self._train_one_step(inputs)
                if self.step % self.train_config.save_steps == 0:
                    tqdm.write(f'[Epoch {epoch+1} | Step {self.step}] Loss {this_loss:.4f}')
                    self.writer.add_scalar('loss', this_loss, self.step)
                    metrics = self.evaluate()
                    metrics_str = '|'.join([f'{k}:{v:.4f}' for k, v in metrics.items()])
                    tqdm.write(f'[Evaluate:{metrics_str}]')

                    if this_loss < self.min_loss:
                        self.min_loss = this_loss
                        tqdm.write('saving the model...')
                        self.model.save_pretrained(self.train_config.output_dir)
                self.step += 1

    # 一次迭代
    def _train_one_step(self, inputs):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        loss_value = outputs.loss
        loss_value.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss_value

    # 验证方法
    def evaluate(self) -> dict:
        dataloader = self._get_dataloader(self.valid_dataset)
        self.model.eval()

        total_loss = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs in tqdm(dataloader, desc='Evaluating: '):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)

                loss_value = outputs.loss
                total_loss += loss_value

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.tolist())

                labels = inputs['labels']
                all_labels.extend(labels.tolist())
            loss = total_loss / len(dataloader)
            metrics = self.compute_metrics(all_labels, all_predictions)
            return {'loss': loss, **metrics}


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
    valid_dataset = get_dataset('valid')
    train_dataset = get_dataset('train')
    collate_fn = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors='pt'
    )
    #6. 评估函数
    def compute_metrics(labels, predictions) -> dict:
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        return {'acc': acc, 'f1': f1}
    #7. 定义训练配置
    train_config = TrainConfig(batch_size=32, output_dir=MODEL_DIR, log_dir=MODEL_DIR, save_steps=100)
    #8. 训练器
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        collate_fn=collate_fn,
        compute_metrics=compute_metrics,
        device=device,
        train_config=train_config
    )
    #9. 训练
    trainer.train()