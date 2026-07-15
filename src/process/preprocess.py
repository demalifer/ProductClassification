from configuration.config import *
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer

#预处理
def preprocess():
    #1. 加载数据集
    dataset_dict = load_dataset(
        'csv',
        data_files={
            'train': str(RAW_DATA_DIR / RAW_TRAIN_DATA),
            'valid': str(RAW_DATA_DIR / RAW_VALID_DATA),
            'test': str(RAW_DATA_DIR / RAW_TEST_DATA)
        },
        delimiter='\t'
    )
    
    #2. 编码
    all_labels = sorted(set(dataset_dict['train']['label']))
    dataset_dict = dataset_dict.cast_column('label', ClassLabel(names=all_labels))
    with open(MODEL_DIR / LABELS_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_labels))

    #3. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    #4. 处理标题文本
    def batch_encode(examples):
        inputs = tokenizer(examples['text_a'], truncation=True)
        inputs['labels'] = examples['label']
        return inputs
    dataset_dict = dataset_dict.map(batch_encode, batched=True, remove_columns=['label', 'text_a'])

    #5. 保存数据集
    dataset_dict.save_to_disk(PROCESSED_DATA_DIR)

if __name__ == '__main__':
    preprocess()