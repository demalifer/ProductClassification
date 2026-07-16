import time
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from torch.optim import Adam

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configuration.config import *
from process.dataset import get_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from torch.amp import GradScaler

# 训练配置类
@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-5
    save_steps: int = 10
    output_dir: str = './models'
    log_dir: str = './logs'
    early_stop_metric: str = 'loss'
    patience: int = 5
    use_amp: bool = True

# 训练器类
class Trainer:
    # 初始化
    def __init__(self, model, train_dataset, valid_dataset, collate_fn, compute_metrics, device, train_config=None):
        # 训练参数
        self.train_config = train_config or TrainConfig()
        # 模型和设备
        self.model = model.to(device)
        self.device = device
        self.use_amp = self.train_config.use_amp and self.device.type == 'cuda'
        # 数据集和数据整理函数
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.collate_fn = collate_fn
        # 评估函数
        self.compute_metrics = compute_metrics
        # 优化器
        self.optimizer = Adam(self.model.parameters(), lr=self.train_config.learning_rate)
        # 全局迭代次数
        self.step = 1
        # tensorboard写入
        self.writer = SummaryWriter(log_dir=str(Path(self.train_config.log_dir) / time.strftime("%Y-%m-%d-%H-%M-%S")))
        # 全局最佳得分
        self.early_stop_best_score = -float('inf')
        # 容忍度计数器
        self.counter = 0
        # AMP梯度缩放器
        self.scaler = GradScaler(device=self.device.type, enabled=self.use_amp)
        # 检查点文件路径
        self.checkpoint_path = Path(self.train_config.output_dir) / 'last' /  'checkpoint.pt'
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    #获取数据加载器
    def _get_dataloader(self, dataset, shuffle=False):
        # 设置格式为tensor
        dataset.set_format(type="torch")

        dataloader = DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn
        )
        return dataloader

    # 核心方法
    def train(self):
        self._load_checkpoint()
        self.model.train()
        # 获取训练集加载器
        dataloader = self._get_dataloader(self.train_dataset, shuffle=True)
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

                    if self._should_stop(metrics):
                        tqdm.write('Early Stopping')
                        return

                    self._save_checkpoint()

                self.step += 1

    # 一次迭代
    def _train_one_step(self, inputs):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_amp
        ):
            outputs = self.model(**inputs)
            loss_value = outputs.loss
        self.scaler.scale(loss_value).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        return loss_value

    # 早停
    def _should_stop(self, metrics):
        metric = float(metrics[self.train_config.early_stop_metric])
        score = -metric if self.train_config.early_stop_metric == 'loss' else metric
        if score > self.early_stop_best_score:
            self.early_stop_best_score = score
            self.counter = 0
            tqdm.write('saving the best model...')
            self.model.save_pretrained(str(Path(self.train_config.output_dir) / 'best'))
            return False
        else:
            self.counter += 1
            if self.counter >= self.train_config.patience:
                return True
            else:
                return False

    # 验证方法
    def evaluate(self) -> dict:
        dataloader = self._get_dataloader(self.valid_dataset, shuffle=False)
        was_training = self.model.training
        self.model.eval()

        total_loss = 0.0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs in tqdm(dataloader, desc='Evaluating: '):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)

                loss_value = outputs.loss
                total_loss += loss_value.item()

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.tolist())

                labels = inputs['labels']
                all_labels.extend(labels.tolist())
        if was_training:
            self.model.train()
        loss = total_loss / len(dataloader)
        metrics = self.compute_metrics(all_labels, all_predictions)
        return {'loss': loss, **metrics}

    #保存检查点
    def _save_checkpoint(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'step': self.step,
            'early_stop_best_score': self.early_stop_best_score,
            'counter': self.counter
        }
        torch.save(checkpoint, self.checkpoint_path)

    #加载检查点
    def _load_checkpoint(self):
        if self.checkpoint_path.exists():
            tqdm.write(f'Loading checkpoint from {self.checkpoint_path}...')
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.step = checkpoint['step']
            self.early_stop_best_score = checkpoint['early_stop_best_score']
            self.counter = checkpoint['counter']
        else:
            tqdm.write(f'No checkpoint found at {self.checkpoint_path}, starting from scratch...')

def train():
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
    train_config = TrainConfig(batch_size=32, output_dir=MODEL_DIR, log_dir=LOG_DIR, save_steps=100)
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

if __name__ == '__main__':
    train()
