from trainer import *
import argparse
import json


def main(args):
    config = json.load(open(args.config))
    assert (config['metric'] in ['acc', 'loss', 'val_loss', 'val_acc'])
    trainer = eval(config['structure'].title()+'Trainer')(config)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("One/Few Shot Learning Implementation")
    parser.add_argument('--config', default='configs/relationnet_config.json')
    main(parser.parse_args())
