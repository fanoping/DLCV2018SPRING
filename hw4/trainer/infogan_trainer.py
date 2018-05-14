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

        self.d_loss_list, self.g_loss_list = [], []
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
                                            shuffle=True)

    def __build_model(self):
        self.g_model = INFOGANGenerator().cuda() if self.with_cuda else INFOGANGenerator()
        self.d_model = INFOGANDiscriminator().cuda() if self.with_cuda else INFOGANDiscriminator()
        self.label_criterion = nn.BCELoss()
        self.class_criterion = nn.BCELoss()
        self.g_optimizer = Adam(self.g_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = Adam(self.d_model.parameters(), lr=0.001, betas=(0.5, 0.999))

    def train(self):
        for epoch in range(1, self.args.epochs + 1):
            g_total_loss, d_total_loss = 0, 0
            real_total_acc, fake_total_acc = 0, 0
            for batch_idx, (in_fig, target_classes) in enumerate(self.train_data_loader):
                batch_size = in_fig.size()[0]
                x, y = Variable(in_fig), Variable(target_classes)
                # labels
                real_labels, fake_labels = Variable(torch.ones(batch_size)), Variable(torch.zeros(batch_size))

                if self.with_cuda:
                    x, y, real_labels, fake_labels = x.cuda(), y.cuda(), real_labels.cuda(), fake_labels.cuda()

                """
                    Train on Discriminator
                """
                # discriminator on real data
                real_output = self.d_model(x)
                real_output = real_output.squeeze()
                real_accuracy = np.mean((real_output[:, 0] > 0.5).cpu().data.numpy())

                # discriminator on fake data
                # discrete noise sample, shape: batch size x 13
                discrete_idx = np.random.randint(10, size=batch_size)
                discrete_noise = np.zeros((batch_size, 10))
                discrete_noise[range(batch_size), discrete_idx] = 1.0
                discrete_noise = torch.FloatTensor(discrete_noise)

                # continuous noise sample
                continuous_noise = torch.randn(batch_size, 1) * 0.5

                # latent noise
                noise = torch.randn(batch_size, 100)

                # fake images
                noise = torch.cat((noise, discrete_noise, continuous_noise), dim=1)
                noise = Variable(noise).cuda() if self.with_cuda else Variable(noise)
                discrete_noise = Variable(discrete_noise).cuda() if self.with_cuda else Variable(discrete_noise)
                image_gen = self.g_model(noise)

                fake_output = self.d_model(image_gen)
                fake_output_c_class, fake_output_d_class = fake_output.squeeze()[:, 1:2], fake_output.squeeze()[:, 2:]
                fake_accuracy = np.mean((fake_output < 0.5).cpu().data.numpy())

                # computing loss
                d_loss_adversarial = -torch.mean(torch.log(real_output) + torch.log(1 - fake_output))
                d_loss_continuous = torch.mean((fake_output_c_class / 0.5) ** 2)
                d_loss_discrete = -torch.mean(torch.sum(discrete_noise * fake_output_d_class, 1)) + \
                                        torch.mean(torch.sum(discrete_noise * discrete_noise, 1))
                d_loss = d_loss_adversarial + 1.0 * d_loss_continuous + 1.0 * d_loss_discrete

                # Back propagation
                self.d_model.zero_grad()
                d_loss.backward(retain_graph=True)
                self.d_optimizer.step()

                """
                    Train on Generator
                """
                g_loss_adversarial = -torch.mean(torch.log(fake_output))
                g_loss = g_loss_adversarial + 1.0 * d_loss_continuous + 1.0 * d_loss_discrete

                # Back propagation
                self.g_model.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                g_total_loss += g_loss.data[0]
                d_total_loss += d_loss.data[0]
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
                    print('Loss: D_loss-{:.6f}'.format(d_loss.data[0]), end=', ')
                    print('G_loss-{:.6f}'.format(g_loss.data[0]), end=', ')
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
            real_ave_acc = real_total_acc / len(self.train_data_loader)
            fake_ave_acc = fake_total_acc / len(self.train_data_loader)

            self.d_loss_list.append(d_ave_loss)
            self.g_loss_list.append(g_ave_loss)
            self.real_acc_list.append(real_ave_acc)
            self.fake_acc_list.append(fake_ave_acc)

            self.__save_checkpoint(epoch)

    def __save_checkpoint(self, epoch):
        state = {
            'model': 'INFOGAN',
            'epoch': epoch,
            'state_dict': [self.g_model.state_dict(), self.d_model.state_dict()],
            'optimizer': [self.g_optimizer.state_dict(), self.d_optimizer.state_dict()],
            'loss': {"d_loss": self.d_loss_list, "g_loss": self.g_loss_list},
            'accuracy': {'real_acc': self.real_acc_list, 'fake_acc': self.fake_acc_list}
        }

        if not os.path.exists("checkpoints/infogan"):
            os.makedirs("checkpoints/infogan")

        filename = "checkpoints/infogan/epoch{}_checkpoint.pth.tar".format(epoch)
        if epoch % self.args.save_freq == 0:
            torch.save(state, f=filename)
