from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn as nn
import torch
from models.rnn_model import RNN
from utils.dataset import TrimmedVideo
import numpy as np
import pickle
import sys
# TODO: args for train
# TODO: save checkpoint


class RNNtrainer:
    def __init__(self, args):
        self.args = args
        self.with_cuda = not self.args.no_cuda
        self.__load_data()
        self.__build_model()

    def __load_data(self):
        self.train_dataset = TrimmedVideo()
        self.train_data_loader = DataLoader(dataset=self.train_dataset,
                                            batch_size=self.args.batch_size,
                                            shuffle=False)
        # valid data
        with open('valid_video.pkl', 'rb') as f:
            self.valid_video = pickle.load(f)
        with open('valid_label.pkl', 'rb') as f:
            self.valid_label = pickle.load(f)
        self.valid_label = np.array(self.valid_label).astype(np.uint8)
        self.valid_label = torch.LongTensor(self.valid_label)

    def __build_model(self):
        self.model = RNN().cuda() if self.with_cuda else RNN()
        self.criterion = nn.CrossEntropyLoss().cuda() if self.with_cuda else nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))

    def train(self):
        self.model.train()
        for epoch in range(1, self.args.epochs+1):
            total_loss, total_acc = 0, 0
            for batch_idx, (video, label) in enumerate(self.train_data_loader):
                video = Variable(video).cuda() if self.with_cuda else Variable(video)
                label = Variable(label).cuda() if self.with_cuda else Variable(label)

                self.optimizer.zero_grad()
                output = self.model(video)
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
                                                                total_acc / len(self.train_data_loader)))

    def valid(self):
        self.model.eval()

        if self.with_cuda:
            valid_video = Variable(self.valid_video).cuda()
            valid_label = Variable(self.valid_label).cuda()
        else:
            valid_video = Variable(self.valid_video)
            valid_label = Variable(self.valid_label)

        output = self.model(valid_video)
        loss = self.criterion(output, valid_label)

        result = torch.max(output, dim=1)[1]
        accuracy = np.mean((result == valid_label).cpu().data.numpy())

        print('valid_loss: {:.6f}  valid_acc: {:.6f}'.format(loss.data[0], accuracy))

