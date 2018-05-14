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
                                            shuffle=True)

    def __build_model(self):
        self.g_model = ACGANGenerator().cuda() if self.with_cuda else ACGANGenerator()
        self.d_model = ACGANDiscriminator().cuda() if self.with_cuda else ACGANDiscriminator()
        self.label_criterion = nn.BCELoss()
        self.class_criterion = nn.BCELoss()
        self.g_optimizer = Adam(self.g_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = Adam(self.d_model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def train(self):
        for epoch in range(1, self.args.epochs + 1):
            g_total_loss, d_total_loss = 0, 0
            real_total_loss, fake_total_loss = 0, 0
            real_total_acc, fake_total_acc = 0, 0
            for batch_idx, (in_fig, target_classes) in enumerate(self.train_data_loader):
                batch_size = in_fig.size()[0]
                x, y = Variable(in_fig), Variable(target_classes[:, 9])  # 9: Smiling
                # labels
                real_labels, fake_labels = Variable(torch.ones(batch_size)), Variable(torch.zeros(batch_size))

                if self.with_cuda:
                    x, y, real_labels, fake_labels = x.cuda(), y.cuda(), real_labels.cuda(), fake_labels.cuda()

                for _ in range(2):
                    """
                        Train on Discriminator
                    """
                    # discriminator on real data
                    real_output, real_output_classes = self.d_model(x)
                    real_output, real_output_classes = real_output.squeeze(), real_output_classes.squeeze()
                    real_label_loss = self.label_criterion(real_output, real_labels)
                    real_classes_loss = self.class_criterion(real_output_classes, y)
                    real_accuracy = np.mean((real_output > 0.5).cpu().data.numpy())
                    real_classes_acc = np.mean((real_output_classes > 0.5).cpu().data.numpy() == y.cpu().data.numpy())
                    real_loss = real_label_loss + real_classes_loss

                    # discriminator on fake data
                    noise = torch.randn(batch_size, 100)
                    random_fake_classes = np.random.randint(2, size=(batch_size, 1))
                    fake_classes = torch.FloatTensor(random_fake_classes)
                    noise = torch.cat((noise, fake_classes), dim=1)
                    noise = Variable(noise).cuda() if self.with_cuda else Variable(noise)
                    fake_classes = Variable(fake_classes).cuda() if self.with_cuda else Variable(fake_classes)
                    image_gen = self.g_model(noise)

                    fake_output, fake_output_classes = self.d_model(image_gen)
                    fake_output, fake_output_classes = fake_output.squeeze(), fake_output_classes.squeeze()
                    fake_label_loss = self.label_criterion(fake_output, fake_labels)
                    fake_classes_loss = self.class_criterion(fake_output_classes, fake_classes.squeeze())
                    fake_accuracy = np.mean((fake_output < 0.5).cpu().data.numpy())
                    fake_loss = fake_label_loss + fake_classes_loss

                    # Back propagation
                    d_loss = real_loss + fake_loss
                    self.d_model.zero_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                """
                    Train on Generator
                """
                noise = torch.randn(batch_size, 100)
                random_fake_classes = np.random.randint(2, size=(batch_size, 1))
                fake_classes = torch.FloatTensor(random_fake_classes)
                noise = torch.cat((noise, fake_classes), dim=1)
                noise = Variable(noise).cuda() if self.with_cuda else Variable(noise)
                fake_classes = Variable(fake_classes).cuda() if self.with_cuda else Variable(fake_classes)
                image_gen = self.g_model(noise)

                fake_output, fake_output_classes = self.d_model(image_gen)
                fake_output, fake_output_classes = fake_output.squeeze(), fake_output_classes.squeeze()
                g_label_loss = self.label_criterion(fake_output, real_labels)
                g_class_loss = self.class_criterion(fake_output_classes, fake_classes.squeeze())

                self.g_model.zero_grad()
                g_loss = g_label_loss + g_class_loss
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
                    print('Loss: D_loss-{:.6f}'.format(d_loss.data[0]), end=', ')
                    print('G_loss-{:.6f}'.format(g_loss.data[0]), end=', ')
                    print('Acc: real-{:.6f}, fake-{:.6f}'.format(
                        real_accuracy,
                        fake_accuracy
                    ), end='\r')
                    sys.stdout.write('\033[K')

            print("Epoch: {}/{} Loss: d_loss-{:.6f}, real_loss-{:.6f}, fake_loss-{:.6f}, g_loss-{:.6f}, "
                  "real_acc-{:.6f}, fake_acc-{:.6f}".format(
                                epoch,
                                self.args.epochs,
                                d_total_loss / len(self.train_data_loader),
                                real_total_loss / len(self.train_data_loader),
                                fake_total_loss / len(self.train_data_loader),
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
            'model': 'ACGAN',
            'epoch': epoch,
            'state_dict': [self.g_model.state_dict(), self.d_model.state_dict()],
            'optimizer': [self.g_optimizer.state_dict(), self.d_optimizer.state_dict()],
            'loss': {"d_loss": self.d_loss_list, "g_loss": self.g_loss_list,
                     'real_loss': self.real_loss_list, 'fake_loss': self.fake_loss_list},
            'accuracy': {'real_acc': self.real_acc_list, 'fake_acc': self.fake_acc_list}
        }

        if not os.path.exists("checkpoints/acgan"):
            os.makedirs("checkpoints/acgan")

        filename = "checkpoints/acgan/epoch{}_checkpoint.pth.tar".format(epoch)
        if epoch % self.args.save_freq == 0:
            torch.save(state, f=filename)
