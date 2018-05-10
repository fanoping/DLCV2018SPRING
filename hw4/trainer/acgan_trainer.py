import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam
from dataset import GANDataset
from models.modules import ACGANGenerator, ACGANDiscriminator
import numpy as np
import torch.nn as nn
import torch
import sys
import os


class ACGANtrainer:
    def __init__(self, args, train_filepath, train_csvfile, test_filepath, test_csvfile):
        self.args = args
        self.with_cuda = not self.args.no_cuda

        self.__load_file(train_filepath, train_csvfile, test_filepath, test_csvfile)
        self.__build_model()

        self.d_loss_list = []
        self.g_loss_list = []

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
                                            shuffle=False)

    def __build_model(self):
        self.g_model = ACGANGenerator().cuda() if self.with_cuda else ACGANGenerator()
        self.d_model = ACGANDiscriminator().cuda() if self.with_cuda else ACGANDiscriminator()
        self.criterion = nn.BCELoss()
        self.g_optimizer = Adam(self.g_model.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.d_optimizer = Adam(self.d_model.parameters(), lr=0.0001, betas=(0.5, 0.999))

    def train(self):
        for epoch in range(1, self.args.epochs + 1):
            g_total_loss, d_real_total_loss, d_fake_total_loss = 0, 0, 0
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
                self.d_model.zero_grad()
                # discriminator on real data
                real_output, real_output_classes = self.d_model(x)
                real_output, real_output_classes = real_output.squeeze(), real_output_classes.squeeze()
                real_loss = self.criterion(real_output, real_labels)
                real_classes_loss = self.criterion(real_output_classes, y)

                # Back propagation
                d_real_loss = real_loss + real_classes_loss
                d_real_loss.backward()

                # discriminator on fake data
                noise = torch.randn(batch_size, 100)
                random_fake_classes = np.random.randint(2, size=(batch_size, 13))
                fake_classes = torch.FloatTensor(random_fake_classes)
                noise = torch.cat((noise, fake_classes), dim=1)
                noise = Variable(noise).cuda() if self.with_cuda else Variable(noise).cuda()
                fake_classes = Variable(fake_classes).cuda() if self.with_cuda else Variable(fake_classes)
                image_gen = self.g_model(noise)

                fake_output, fake_output_classes = self.d_model(image_gen)
                fake_output, fake_output_classes = fake_output.squeeze(), fake_output_classes.squeeze()
                fake_loss = self.criterion(fake_output, fake_labels)
                fake_classes_loss = self.criterion(fake_output_classes, fake_classes.squeeze())

                # Back propagation
                d_fake_loss = fake_loss + fake_classes_loss
                d_fake_loss.backward()
                self.d_optimizer.step()

                """
                    Train on Generator
                """

                noise = torch.randn(batch_size, 100)
                random_fake_classes = np.random.randint(2, size=(batch_size, 13))
                fake_classes = torch.FloatTensor(random_fake_classes)
                noise = torch.cat((noise, fake_classes), dim=1)
                noise = Variable(noise).cuda() if self.with_cuda else Variable(noise).cuda()
                image_gen = self.g_model(noise)

                fake_output, fake_output_classes = self.d_model(image_gen)
                fake_output, fake_output_classes = fake_output.squeeze(), fake_output_classes.squeeze()
                g_label_loss = self.criterion(fake_output, real_labels)
                g_class_loss = self.criterion(fake_output_classes, y)
                g_loss = g_label_loss + g_class_loss

                self.d_model.zero_grad()
                self.g_model.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                g_total_loss += g_loss.data[0]
                d_real_total_loss += d_real_loss.data[0]
                d_fake_total_loss += d_fake_loss.data[0]

                if batch_idx % self.args.log_step == 0:
                    print('Epoch: {}/{} [{}/{} ({:.0f}%)]'.format(
                        epoch,
                        self.args.epochs,
                        batch_idx * self.train_data_loader.batch_size,
                        len(self.train_data_loader) * self.train_data_loader.batch_size,
                        100.0 * batch_idx / len(self.train_data_loader)
                    ), end=' ')
                    print('Loss: D_loss(rl:{:.6f},fl:{:.6f}))'.format(
                        d_real_loss.data[0],
                        d_fake_loss.data[0],
                    ), end=' ')
                    print('|', end=' ')
                    print('G_loss(gl:{:.6f}, gcl:{:.6f}, t:{:.6f})'.format(
                        g_label_loss.data[0],
                        g_class_loss.data[0],
                        g_loss.data[0]
                    ), end='\r')
                    sys.stdout.write('\033[K')

            print("Epoch: {}/{} Loss: d_loss-{:.6f},{:.6f}, g_loss-{:.6f}".format(epoch,
                                                                                  self.args.epochs,
                                                                                  d_real_total_loss / len(
                                                                                      self.train_data_loader),
                                                                                  d_fake_total_loss / len(
                                                                                      self.train_data_loader),
                                                                                  g_total_loss / len(
                                                                                      self.train_data_loader)))

            g_ave_loss = g_total_loss / len(self.train_data_loader)
            d_real_ave_loss = d_real_loss / len(self.train_data_loader)
            d_fake_ave_loss = d_fake_loss / len(self.train_data_loader)

            self.d_loss_list.append((d_real_ave_loss, d_fake_ave_loss))
            self.g_loss_list.append(g_ave_loss)

            self.__save_checkpoint(epoch)

    def __save_checkpoint(self, epoch):
        state = {
            'model': 'ACGAN',
            'epoch': epoch,
            'state_dict': [self.g_model.state_dict(), self.d_model.state_dict()],
            'optimizer': [self.g_optimizer.state_dict(), self.d_optimizer.state_dict()],
            'loss': {"d_loss": self.d_loss_list, "g_loss": self.g_loss_list}
        }

        if not os.path.exists("checkpoints/acgan"):
            os.makedirs("checkpoints/acgan")

        filename = "checkpoints/acgan/epoch{}_checkpoint.pth.tar".format(epoch)
        if epoch % self.args.save_freq == 0:
            torch.save(state, f=filename)
