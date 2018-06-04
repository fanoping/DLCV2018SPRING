from models.modules.pretrained import Densenet121, Resnet50, Vgg19
from utils.reader import getVideoList, getLabelList
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import random


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
            force = self.args.force
            if os.path.exists('cnn_train_feature.tar') and not force:
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
            force = self.args.force
            if os.path.exists('cnn_valid_feature.tar') and not force:
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
            if os.path.exists(self.args.input_feature):
                self.valid_feature = torch.load(self.args.input_feature)
            else:
                return NotImplementedError('Haven\'t trained the model yet!')

            self.valid_label = getVideoList(self.args.input_csv)['Action_labels']
            self.valid_label = np.array(self.valid_label).astype(np.uint8)
            self.valid_label = torch.LongTensor(self.valid_label)

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


class FullLengthVideo(Dataset):
    def __init__(self, args, mode='train'):
        super(FullLengthVideo, self).__init__()
        self.args = args
        self.mode = mode
        self.with_cuda = not self.args.no_cuda
        self.pretrained = eval(args.pretrained.title())().cuda() if self.with_cuda else eval(args.pretrained.title())()
        print("Using {} pretrained model.".format(args.pretrained))

        self.__load_data()

    def __load_data(self):
        print("Loading {} data".format(self.mode))

        if self.mode == 'train':
            force = self.args.force
            if os.path.exists('rnn_full_length_train_feature.tar') and not force:
                self.train_feature = torch.load('rnn_full_length_train_feature.tar')
            else:
                train_video = torch.load('train_full_length_video.tar')

                with torch.no_grad():
                    self.pretrained.eval()

                    self.train_feature = []
                    for _, frames in train_video.items():
                        video = []
                        frames = frames.cuda() if self.with_cuda else frames
                        for i in range(len(frames)):
                            video.append(self.pretrained(frames[i].unsqueeze(0)).squeeze(0))
                        self.train_feature.append(torch.stack(video))

                    torch.save(self.train_feature, 'rnn_full_length_train_feature.tar')

            self.train_label = getLabelList('HW5_data/FullLengthVideos/labels/train')
            self.train_label = [torch.LongTensor(labels) for labels in self.train_label]

        elif self.mode == 'valid':
            force = self.args.force
            if os.path.exists('rnn_full_length_valid_feature.tar') and not force:
                self.valid_feature = torch.load('rnn_full_length_valid_feature.tar')
            else:
                valid_video = torch.load('valid_full_length_video.tar')

                self.valid_feature = []
                with torch.no_grad():
                    self.pretrained.eval()

                    for _, frames in valid_video.items():
                        video = []
                        frames = frames.cuda() if self.with_cuda else frames
                        for i in range(len(frames)):
                            video.append(self.pretrained(frames[i].unsqueeze(0)).squeeze(0))
                        self.valid_feature.append(torch.stack(video))

                    torch.save(self.valid_feature, 'rnn_full_length_valid_feature.tar')

            self.valid_label = getLabelList('HW5_data/FullLengthVideos/labels/valid')
            self.valid_label = [torch.LongTensor(labels) for labels in self.valid_label]

        else:
            if os.path.exists(self.args.input_feature):
                self.valid_feature = torch.load(self.args.input_feature)
            else:
                return NotImplementedError('Haven\'t trained the model yet!')

            self.valid_label = getLabelList(self.args.input_txt)
            self.valid_label = [torch.LongTensor(labels) for labels in self.valid_label]

    def rnn_collate_fn(self, batch):
        if self.mode == 'eval':
            data, label = [], []
            for frame, item in batch:
                data.append(frame.view(frame.size(0), -1))
                label.append(item)
            data = pad_sequence(data)
            length = [frame.size(0) for frame in data]
            label = torch.stack(label)
            label = label.transpose(0, 1)
        else:
            max_sample = 512
            data, label = [], []
            for frame, item in batch:
                selected_idx = sorted(random.sample([i for i in range(0, frame.size(0))], max_sample))
                data.append(frame.view(frame.size(0), -1)[selected_idx])
                label.append(item[selected_idx])
            data = pad_sequence(data)
            length = [frame.size(0) for frame in data]
            label = torch.stack(label)
            label = label.transpose(0, 1)

        return [data, label, length]

    def __getitem__(self, index):
        video = self.train_feature[index] if self.mode == 'train' else self.valid_feature[index]
        label = self.train_label[index] if self.mode == 'train' else self.valid_label[index]

        return video, label

    def __len__(self):
        return len(self.train_feature) if self.mode == 'train' else len(self.valid_feature)
