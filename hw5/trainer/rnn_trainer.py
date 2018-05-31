from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam, RMSprop
import torch.nn as nn
import torch
from models.rnn_model import RNN
from utils.dataset import TrimmedVideo
import numpy as np
import sys
import os


class RNNtrainer:
    def __init__(self, args):
        self.args = args
        self.with_cuda = not self.args.no_cuda
        self.__load_data()
        self.__build_model()

        self.loss_list, self.val_loss_list = [], []
        self.acc_list, self.val_acc_list = [], []
        self.max_acc = 0

    def __load_data(self):
        self.train_dataset = TrimmedVideo(self.args, 'train')
        self.train_data_loader = DataLoader(dataset=self.train_dataset,
                                            batch_size=self.args.batch_size,
                                            collate_fn=self.train_dataset.rnn_collate_fn,
                                            shuffle=True)

        self.valid_dataset = TrimmedVideo(self.args, 'valid')
        self.valid_data_loader = DataLoader(dataset=self.valid_dataset,
                                            batch_size=self.args.batch_size,
                                            collate_fn=self.valid_dataset.rnn_collate_fn,
                                            shuffle=False)

    def __build_model(self):
        self.model = RNN(self.args).cuda() if self.with_cuda else RNN(self.args)
        self.criterion = nn.CrossEntropyLoss().cuda() if self.with_cuda else nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.0001)

    def train(self):
        for epoch in range(1, self.args.epochs+1):
            self.model.train()
            total_loss, total_acc = 0, 0
            for batch_idx, (video, label, length) in enumerate(self.train_data_loader):
                video = Variable(video).cuda() if self.with_cuda else Variable(video)
                label = Variable(label).cuda() if self.with_cuda else Variable(label)

                self.optimizer.zero_grad()
                output = self.model(video, length)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()

                result = torch.max(output, dim=1)[1]
                accuracy = np.mean((result == label).cpu().data.numpy())

                total_loss += loss.data[0]
                total_acc += accuracy
                if batch_idx % self.args.log_step == 0:
                    print('Epoch: {}/{} [{}/{} ({:.0f}%)] loss: {:.6f}, acc: {:.6f}'.format(
                            epoch,
                            self.args.epochs,
                            batch_idx * self.train_data_loader.batch_size,
                            len(self.train_data_loader) * self.train_data_loader.batch_size,
                            100.0 * batch_idx / len(self.train_data_loader),
                            loss.data[0],
                            accuracy
                    ), end='\r')
                    sys.stdout.write('\033[K')

            print("Epoch: {}/{} loss:{:.6f}  acc:{:.6f}".format(epoch,
                                                                self.args.epochs,
                                                                total_loss / len(self.train_data_loader),
                                                                total_acc / len(self.train_data_loader)), end=' ')

            ave_loss = total_loss / len(self.train_data_loader)
            ave_acc = total_acc / len(self.train_data_loader)
            self.loss_list.append(ave_loss)
            self.acc_list.append(ave_acc)

            if self.args.verbosity == 1:
                _, val_acc = self.valid()
                self.__save_checkpoint(epoch, val_acc)
            else:
                print()
                self.__save_checkpoint(epoch, ave_acc)

    def valid(self):
        with torch.no_grad():
            self.model.eval()
            total_loss, total_acc = 0, 0
            for batch_idx, (video, label, length) in enumerate(self.valid_data_loader):
                video = Variable(video).cuda() if self.with_cuda else Variable(video)
                label = Variable(label).cuda() if self.with_cuda else Variable(label)

                output = self.model(video, length)
                loss = self.criterion(output, label)

                result = torch.max(output, dim=1)[1]
                accuracy = np.mean((result == label).cpu().data.numpy())

                total_loss += loss.data[0]
                total_acc += accuracy

            print('valid_loss: {:.6f}  valid_acc: {:.6f}'.format(total_loss / len(self.valid_data_loader),
                                                                 total_acc / len(self.valid_data_loader)))

            self.val_loss_list.append(total_loss / len(self.valid_data_loader))
            self.val_acc_list.append(total_acc / len(self.valid_data_loader))

        return total_loss / len(self.valid_data_loader), total_acc / len(self.valid_data_loader)

    def __save_checkpoint(self, epoch, current_acc=None):
        state = {
            'model': '{}-rnn-classifier'.format(self.args.pretrained),
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': self.loss_list,
            'accuracy': self.acc_list,
            'val_loss': self.val_loss_list,
            'val_accuracy': self.val_acc_list
        }

        filepath = os.path.join("checkpoints", "rnn_{}".format(self.args.pretrained.lower()))
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        filename = os.path.join(filepath, "epoch{}_checkpoint.pth.tar".format(epoch))
        if epoch % self.args.save_freq == 0:
            torch.save(state, f=filename)

        best_filename = os.path.join(filepath, "best_checkpoint.pth.tar")
        if self.max_acc < current_acc:
            torch.save(state, f=best_filename)
            print("Saving Epoch: {}, Updating acc {:.6f} to {:.6f}".format(epoch, self.max_acc, current_acc))
            self.max_acc = current_acc

