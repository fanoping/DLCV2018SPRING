from trainer import *
import argparse
import json


def main(args):
    config = json.load(open(args.config))
    trainer = eval(config['structure'].title()+'Trainer')(config)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Task 2")
    parser.add_argument('--config', default='configs/protonet_config.json')
    main(parser.parse_args())
