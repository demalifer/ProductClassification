from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from configuration.config import *

def get_dataset(ds_type='train'):
    path = str(PROCESSED_DATA_DIR / ds_type)
    dataset = load_from_disk(path)
    return dataset

# 获取数据加载器
def get_dataloader(tokenizer, ds_type='train'):
    #1. 加载数据集
    path = str(PROCESSED_DATA_DIR / ds_type)
    dataset = load_from_disk(path)

    #2. 设置格式为tensor
    dataset.set_format(type='torch')

    #3. 创建DataLoader
    collate_fn = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors='pt',
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    return dataloader

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    dataloader = get_dataloader(tokenizer)

    for batch in dataloader:
        for key, value in batch.items():
            print(key, value)
        break