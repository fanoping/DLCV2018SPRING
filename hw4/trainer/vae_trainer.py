import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam
from dataset import CelebADataset
from models.modules import AE, VAE
from models.loss import CustomLoss
import torch
import sys
import os


class VAEtrainer:
    def __init__(self, args, train_filepath, train_csvfile, test_filepath, test_csvfile):
        self.args = args
        self.with_cuda = not self.args.no_cuda

        self.__load_file(train_filepath, train_csvfile, test_filepath, test_csvfile)
        self.__build_model()

        self.min_loss = float('inf')
        self.kld_loss_list, self.mse_loss_list = [], []

    def __load_file(self, train_filepath, train_csvfile, test_filepath, test_csvfile):
        self.train_dataset = CelebADataset(train_filepath,
                                      train_csvfile,
                                      test_filepath,
                                      test_csvfile,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                      ]))
        self.train_data_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=self.args.batch_size,
                                       shuffle=False)

    def __build_model(self):
        self.model = VAE().cuda() if self.with_cuda else VAE()
        self.criterion = CustomLoss(5e-7)
        self.optimizer = Adam(self.model.parameters(), lr=0.0005)

    def train(self):
        for epoch in range(1, self.args.epochs + 1):
            mse_loss, kld_loss, total_loss = 0, 0, 0
            for batch_idx, (in_fig, _) in enumerate(self.train_data_loader):
                x = Variable(in_fig).cuda() if self.with_cuda else Variable(in_fig)

                self.optimizer.zero_grad()
                output, mu, logvar = self.model(x)
                loss = self.criterion(output, x, mu, logvar)
                loss.backward()
                self.optimizer.step()

                mse_loss += self.criterion.latestloss()['MSE'].data[0]
                kld_loss += self.criterion.latestloss()['KLD'].data[0]
                total_loss += loss.data[0]
                if batch_idx % self.args.log_step == 0:
                    print('Epoch: {}/{} [{}/{} ({:.0f}%)] Loss: mse({:.6f}), kld({:.6f})'.format(
                        epoch,
                        self.args.epochs,
                        batch_idx * self.train_data_loader.batch_size,
                        len(self.train_data_loader) * self.train_data_loader.batch_size,
                        100.0 * batch_idx / len(self.train_data_loader),
                        self.criterion.latestloss()['MSE'].data[0],
                        self.criterion.latestloss()['KLD'].data[0]
                    ), end='\r')
                    sys.stdout.write('\033[K')
            print("Epoch: {}/{} Loss:{:.6f}".format(epoch, self.args.epochs, total_loss / len(self.train_data_loader)))

            self.mse_loss_list.append(mse_loss / len(self.train_data_loader))
            self.kld_loss_list.append(kld_loss / len(self.train_data_loader))
            self.__save_checkpoint(epoch, total_loss / len(self.train_data_loader))

    def __save_checkpoint(self, epoch, current_loss):
        state = {
            'model': 'VAE',
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'kld_loss': self.kld_loss_list,
            'mse_loss': self.mse_loss_list
        }

        if not os.path.exists("checkpoints/vae"):
            os.makedirs("checkpoints/vae")

        filename = "checkpoints/vae/epoch{}_checkpoint.pth.tar".format(epoch)
        best_filename = "checkpoints/vae/best_checkpoint.pth.tar"

        if epoch % self.args.save_freq == 0:
            torch.save(state, f=filename)
        if self.min_loss > current_loss:
            torch.save(state, f=best_filename)
            print("Saving Epoch: {}, Updating loss {:.6f} to {:.6f}".format(epoch, self.min_loss, current_loss))
            self.min_loss = current_loss

