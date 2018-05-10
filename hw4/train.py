from trainer.vae_trainer import VAEtrainer
from trainer.gan_trainer import GANtrainer
from trainer.acgan_trainer import ACGANtrainer
import argparse


def main(args):
    train_data_path = 'hw4_data/train'
    train_data_csv = 'hw4_data/train.csv'
    test_data_path = 'hw4_data/test'
    test_data_csv = 'hw4_data/test.csv'

    if args.arch == 'VAE':
        trainer = VAEtrainer(args, train_data_path, train_data_csv, test_data_path, test_data_csv)
    elif args.arch == 'GAN':
        trainer = GANtrainer(args, train_data_path, train_data_csv, test_data_path, test_data_csv)
    else:
        trainer = ACGANtrainer(args, train_data_path, train_data_csv, test_data_path, test_data_csv)

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train")
    parser.add_argument('--arch', default='ACGAN', type=str,
                        help='training architecture [AE, VAE, GAN, ACGAN, InfoGAN]')
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
