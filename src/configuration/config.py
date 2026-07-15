from pathlib import Path

#1. 目录路径
ROOT_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = ROOT_DIR / 'data'/ 'raw'
PROCESSED_DATA_DIR = ROOT_DIR / 'data'/ 'processed'
MODEL_DIR = ROOT_DIR / 'models'
LOG_DIR = ROOT_DIR / 'logs'

#2. 文件
RAW_TRAIN_DATA = 'train.txt'
RAW_VALID_DATA = 'valid.txt'
RAW_TEST_DATA = 'test.txt'
BERT_MODEL_NAME = 'google-bert/bert-base-chinese'
LABELS_FILE = 'labels.txt'

#3. 超参数
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
EPOCHS = 10
SAVE_STEPS = 50