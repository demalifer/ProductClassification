from argparse import ArgumentParser
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main():
    parser = ArgumentParser()
    parser.add_argument('action', choices=['preprocess','train', 'predict', 'evaluate', 'serve'])
    args = parser.parse_args()
    action = args.action

    match action:
        case 'preprocess':
            from process.preprocess import preprocess
            preprocess()
        case 'train':
            from runner.train import train
            train()
        case 'predict':
            from runner.predict import predict
            predict()
        case 'evaluate':
            from runner.evaluate import evaluate
            evaluate()
        case 'serve':
            from web.app import serve
            serve()


if __name__ == '__main__':
    main()
