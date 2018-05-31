from models.modules.pretrained import Densenet121, Resnet50, Vgg19
from utils.reader import getVideoList
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os


class TrimmedVideo(Dataset):
    def __init__(self, args, mode='train'):
        super(TrimmedVideo, self).__init__()
        self.args = args
        self.mode = mode
        self.with_cuda = not self.args.no_cuda
        self.pretrained = eval(args.pretrained.title())().cuda() if self.with_cuda else eval(args.pretrained.title())()
        print("Using {} pretrained model.".format(args.pretrained))

        self.__load_data()

    def __load_data(self):
        print("Loading {} data".format(self.mode))
        if self.mode == 'train':

            if os.path.exists('cnn_train_feature.tar') and not self.args.force:
                self.train_feature = torch.load('cnn_train_feature.tar')
            else:
                train_video = torch.load('train_video.tar')

                with torch.no_grad():
                    self.pretrained.eval()
                    self.train_feature = [self.pretrained(frames.cuda()) for _, frames in train_video.items()]
                    torch.save(self.train_feature, 'cnn_train_feature.tar')


            self.train_label = getVideoList('HW5_data/TrimmedVideos/label/gt_train.csv')['Action_labels']
            self.train_label = np.array(self.train_label).astype(np.uint8)
            self.train_label = torch.LongTensor(self.train_label)

        elif self.mode == 'valid':
            if os.path.exists('cnn_valid_feature.tar') and not self.args.force:
                self.valid_feature = torch.load('cnn_valid_feature.tar')
            else:
                valid_video = torch.load('valid_video.tar')

                with torch.no_grad():
                    self.pretrained.eval()
                    self.valid_feature = [self.pretrained(frames.cuda()) for _, frames in valid_video.items()]
                    torch.save(self.valid_feature, 'cnn_valid_feature.tar')

            self.valid_label = getVideoList('HW5_data/TrimmedVideos/label/gt_valid.csv')['Action_labels']
            self.valid_label = np.array(self.valid_label).astype(np.uint8)
            self.valid_label = torch.LongTensor(self.valid_label)

        else:
            if os.path.exists('cnn_valid_feature.tar'):
                self.valid_feature = torch.load('cnn_valid_feature.tar')
            else:
                return NotImplementedError('Haven\'t trained the model yet!')

    def cnn_collate_fn(self, batch):
        batch = [(torch.mean(frame, dim=0), label) for frame, label in batch]
        data = [frame.view(1, frame.size(0)) for frame, _ in batch]
        data = torch.cat(data, 0)
        label = [item for _, item in batch]
        label = torch.LongTensor(label)

        return [data, label]

    def rnn_collate_fn(self, batch):
        batch = sorted(batch, key=lambda k: k[0].size(0))[::-1]
        data = [frame.view(frame.size(0), -1) for frame, _ in batch]
        length = [frame.size(0) for frame in data]
        data = pad_sequence(data)
        label = [item for _, item in batch]
        label = torch.LongTensor(label)

        return [data, label, length]

    def __getitem__(self, index):
        video = self.train_feature[index] if self.mode == 'train' else self.valid_feature[index]
        label = self.train_label[index] if self.mode == 'train' else self.valid_label[index]

        return video, label

    def __len__(self):
        return len(self.train_feature) if self.mode == 'train' else len(self.valid_feature)
