import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam
from dataset import GANDataset
from models.modules import INFOGANGenerator, INFOGANDiscriminator
import numpy as np
import torch.nn as nn
import torch
import sys
import os


class INFOGANtrainer:
    def __init__(self, args, train_filepath, train_csvfile, test_filepath, test_csvfile):
        self.args = args
        self.with_cuda = not self.args.no_cuda

        self.__load_file(train_filepath, train_csvfile, test_filepath, test_csvfile)
        self.__build_model()

        self.latent_dim = 128
        self.discrete_cat_dim = 10

        self.d_loss_list, self.g_loss_list = [], []
        self.real_loss_list, self.fake_loss_list = [], []
        self.real_acc_list, self.fake_acc_list = [], []

    def __load_file(self, train_filepath, train_csvfile, test_filepath, test_csvfile):
        self.train_dataset = GANDataset(train_filepath,
                                        train_csvfile,
                                        test_filepath,
                                        test_csvfile,
                                        transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                        ]))
        self.train_data_loader = DataLoader(dataset=self.train_dataset,
                                            batch_size=self.args.batch_size,
                                            shuffle=True,
                                            num_workers=1)

    def __build_model(self):
        self.g_model = INFOGANGenerator().cuda() if self.with_cuda else INFOGANGenerator()
        self.d_model = INFOGANDiscriminator().cuda() if self.with_cuda else INFOGANDiscriminator()

        self.d_criterion = nn.BCELoss().cuda() if self.with_cuda else nn.BCELoss()
        self.q_discrete_criterion = nn.CrossEntropyLoss().cuda() if self.with_cuda else nn.CrossEntropyLoss()
        self.q_continuous_criterion = nn.MSELoss().cuda() if self.with_cuda else nn.MSELoss()

        self.g_optimizer = Adam(self.g_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = Adam(self.d_model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def train(self):
        for epoch in range(1, self.args.epochs + 1):
            g_total_loss, d_total_loss = 0, 0
            real_total_loss, fake_total_loss = 0, 0
            real_total_acc, fake_total_acc = 0, 0
            for batch_idx, (in_fig, _) in enumerate(self.train_data_loader):
                batch_size = in_fig.size()[0]
                x = Variable(in_fig)
                # labels
                real_labels, fake_labels = Variable(torch.ones(batch_size, 1)), Variable(torch.zeros(batch_size, 1))

                if self.with_cuda:
                    x, real_labels, fake_labels = x.cuda(), real_labels.cuda(), fake_labels.cuda()

                """
                    Train on Discriminator
                """
                self.d_model.zero_grad()
                # discriminator on real data
                real_output, _ = self.d_model(x)
                real_loss = self.d_criterion(real_output, real_labels)
                real_accuracy = np.mean((real_output > 0.5).cpu().data.numpy())

                # discriminator on fake data
                # discrete noise sample, shape: batch size x (10)
                discrete = torch.FloatTensor()
                for _ in range(10):
                    discrete_idx = np.random.randint(10, size=batch_size)
                    discrete_noise = np.zeros((batch_size, 10))
                    discrete_noise[range(batch_size), discrete_idx] = 1.0
                    discrete_noise = torch.FloatTensor(discrete_noise)
                    discrete = torch.cat((discrete, discrete_noise), dim=1)
                discrete_noise = Variable(discrete)

                # latent noise
                noise = torch.randn(batch_size, self.latent_dim)
                noise = Variable(noise)

                # fake images
                noise = torch.cat((noise, discrete_noise), dim=1)
                noise = noise.cuda() if self.with_cuda else noise
                image_gen = self.g_model(noise)

                fake_output, _ = self.d_model(image_gen)
                fake_loss = self.d_criterion(fake_output, fake_labels)
                fake_accuracy = np.mean((fake_output < 0.5).cpu().data.numpy())

                # compute loss
                d_loss = real_loss + fake_loss
                d_loss.backward()
                self.d_optimizer.step()

                """
                    Train on Generator
                """
                self.g_model.zero_grad()

                image_gen = self.g_model(noise)
                fake_output, fake_discrete = self.d_model(image_gen)
                g_recon_loss = self.d_criterion(fake_output, real_labels)

                discrete_noise = discrete_noise.cuda() if self.with_cuda else discrete_noise
                discrete_loss = 0
                for idx in range(10):
                    discrete_loss += self.q_discrete_criterion(fake_discrete[idx*10:(idx+1)*10],
                                                               torch.max(discrete_noise[idx*10:(idx+1)*10], 1)[1])

                # generator loss
                g_loss = g_recon_loss + 0.1 * discrete_loss

                # Back propagation
                g_loss.backward()
                self.g_optimizer.step()

                g_total_loss += g_loss.data[0]
                d_total_loss += d_loss.data[0]
                real_total_loss += real_loss.data[0]
                fake_total_loss += fake_loss.data[0]
                real_total_acc += real_accuracy
                fake_total_acc += fake_accuracy

                if batch_idx % self.args.log_step == 0:
                    print('Epoch: {}/{} [{}/{} ({:.0f}%)]'.format(
                        epoch,
                        self.args.epochs,
                        batch_idx * self.train_data_loader.batch_size,
                        len(self.train_data_loader) * self.train_data_loader.batch_size,
                        100.0 * batch_idx / len(self.train_data_loader)
                    ), end=' ')
                    print('Loss: D_loss-{:.6f}, {:.6f}'.format(real_loss.data[0], fake_loss.data[0]), end=', ')
                    print('G_loss-{:.6f}, {:.6f}'.format(g_recon_loss.data[0], discrete_loss.data[0]), end=', ')
                    print('Acc: real-{:.6f}, fake-{:.6f}'.format(
                        real_accuracy,
                        fake_accuracy
                    ), end='\r')
                    sys.stdout.write('\033[K')

            print("Epoch: {}/{} Loss: d_loss-{:.6f}, g_loss-{:.6f}, "
                  "real_acc-{:.6f}, fake_acc-{:.6f}".format(
                                epoch,
                                self.args.epochs,
                                d_total_loss / len(self.train_data_loader),
                                g_total_loss / len(self.train_data_loader),
                                real_total_acc / len(self.train_data_loader),
                                fake_total_acc / len(self.train_data_loader)
                  ))

            g_ave_loss = g_total_loss / len(self.train_data_loader)
            d_ave_loss = d_total_loss / len(self.train_data_loader)
            real_ave_loss = real_total_loss / len(self.train_data_loader)
            fake_ave_loss = fake_total_loss / len(self.train_data_loader)
            real_ave_acc = real_total_acc / len(self.train_data_loader)
            fake_ave_acc = fake_total_acc / len(self.train_data_loader)

            self.d_loss_list.append(d_ave_loss)
            self.g_loss_list.append(g_ave_loss)
            self.real_loss_list.append(real_ave_loss)
            self.fake_loss_list.append(fake_ave_loss)
            self.real_acc_list.append(real_ave_acc)
            self.fake_acc_list.append(fake_ave_acc)

            self.__save_checkpoint(epoch)

    def __save_checkpoint(self, epoch):
        state = {
            'model': 'INFOGAN',
            'epoch': epoch,
            'state_dict': [self.g_model.state_dict(), self.d_model.state_dict()],
            'optimizer': [self.g_optimizer.state_dict(), self.d_optimizer.state_dict()],
            'loss': {"d_loss": self.d_loss_list, "g_loss": self.g_loss_list,
                     "real_loss": self.real_loss_list, "fake_loss": self.fake_loss_list},
            'accuracy': {'real_acc': self.real_acc_list, 'fake_acc': self.fake_acc_list}
        }

        if not os.path.exists("checkpoints/infogan"):
            os.makedirs("checkpoints/infogan")

        filename = "checkpoints/infogan/epoch{}_checkpoint.pth.tar".format(epoch)
        if epoch % self.args.save_freq == 0:
            torch.save(state, f=filename)
