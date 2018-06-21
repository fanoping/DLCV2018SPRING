from trainer import FewshotTrainer
import argparse
import json


def main(args):
    trainer = FewshotTrainer(json.load(open(args.config)))
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Task 2")
    parser.add_argument('--config', default='config.json')
    main(parser.parse_args())
