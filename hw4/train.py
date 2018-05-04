from dataset import CelebADataset
from models.modules import AE, VAE
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
from models.loss import CustomLoss


def main(args):
    # load data
    train_data_path = 'hw4_data/train'
    train_dataset = CelebADataset(train_data_path)
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=False)
    # gpu configuration
    with_cuda = not args.no_cuda

    # model construction
    if args.arch == "AE":
        model, criterion = AE(), nn.MSELoss()
    elif args.arch == "VAE":
        model, criterion = VAE(), CustomLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    print(model)
    if with_cuda:
        model = model.cuda()

    # training
    for epoch in range(1, args.epochs+1):
        total_loss = 0
        for batch_idx, (in_fig, target_fig) in enumerate(train_data_loader):
            x, y = Variable(in_fig), Variable(target_fig)
            if with_cuda:
                x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            output, mu, logvar = model(x)
            # Should be fixed
            loss = criterion(output, y, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.data[0]
            if batch_idx % args.log_step == 0:
                print('Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * train_data_loader.batch_size,
                    len(train_data_loader) * train_data_loader.batch_size,
                    100.0 * batch_idx / len(train_data_loader),
                    loss.data[0]))
        print("Epoch:{} Loss:{}".format(epoch, total_loss / len(train_data_loader)))

        state = {
            'model': args.arch,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        filename = "checkpoints/epoch{}_checkpoint.pth.tar".format(epoch)
        torch.save(state, f=filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train")
    parser.add_argument('--arch', default='AE', type=str,
                        help='training architecture [AE, VAE, GAN, ACGAN, InfoGAN]')
    parser.add_argument('--batch-size', default=8, type=int,
                        help='batch size of the model (default: 8)')
    parser.add_argument('--epochs', default=10, type=int,
                        help='training epochs (default: 1)')
    parser.add_argument('--log-step', default=500, type=int,
                        help='printing step size (default: 500')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use CPU in case there\'s no GPU support')
    main(parser.parse_args())
