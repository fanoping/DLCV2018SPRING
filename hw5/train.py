from trainer.cnn_trainer import CNNtrainer
from trainer.rnn_trainer import RNNtrainer
import argparse


def main(args):
    trainer = eval(args.arch.upper() + 'trainer')(args)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Homework 4")
    parser.add_argument('--arch', default='CNN', type=str,
                        help='training architecture [CNN, RNN]')
    parser.add_argument('--pretrained', default='Resnet50', type=str,
                        help='training architecture [Vgg19, Resnet50]')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='batch size of the model (default: 32)')
    parser.add_argument('--epochs', default=100, type=int,
                        help='training epochs (default: 100)')
    parser.add_argument('--log-step', default=1, type=int,
                        help='printing step size (default: 1')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--save-freq', default=1, type=int,
                        help='save checkpoints frequency (default: 1)')
    parser.add_argument('--verbosity', default=1, type=int,
                        help='verbosity 0 or 1 (default: 1)')
    parser.add_argument('-f', '--force', action='store_true',
                        help='force changing the pretrained feature file')
    main(parser.parse_args())