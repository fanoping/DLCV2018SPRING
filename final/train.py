from trainer.supervised_trainer import SupervisedTrainer
import argparse


def main(args):
    trainer = eval(args.arch.title() + 'Trainer')(args)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Final Project")
    parser.add_argument('--arch', default='Supervised', type=str,
                        help='training architecture [Supervised]')
    parser.add_argument('--train-dir', default='datasets/Fashion_MNIST_student/train',
                        help='training data directory')
    parser.add_argument('--test-dir', default='datasets/Fashion_MNIST_student/test',
                        help='testing data directory')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='batch size of the model (default: 128)')
    parser.add_argument('--epochs', default=100, type=int,
                        help='training epochs (default: 100)')
    parser.add_argument('--log-step', default=1, type=int,
                        help='printing step size (default: 10')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--save-freq', default=1, type=int,
                        help='save checkpoints frequency (default: 1)')
    main(parser.parse_args())
