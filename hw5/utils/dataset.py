from utils.reader import getVideoList
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np
import pickle


class TrimmedVideo(Dataset):
    def __init__(self, mode='train'):
        super(TrimmedVideo, self).__init__()
        self.with_cuda = True
        self.mode = mode
        self.__load_data()

    def __load_data(self):
        print("Loading {} data".format(self.mode))
        if self.mode == 'train':
            with open('train_video.pkl', 'rb') as f:
                self.train_video = pickle.load(f)

            self.train_label = getVideoList('HW5_data/TrimmedVideos/label/gt_train.csv')['Action_labels']
            self.train_label = np.array(self.train_label).astype(np.uint8)
            self.train_label = torch.from_numpy(self.train_label).type(torch.long)
        else:
            with open('valid_video.pkl', 'rb') as f:
                self.valid_video = pickle.load(f)

            self.valid_label = getVideoList('HW5_data/TrimmedVideos/label/gt_train.csv')['Action_labels']
            self.valid_label = np.array(self.valid_label).astype(np.uint8)
            self.valid_label = torch.from_numpy(self.valid_label).type(torch.long)

    def __getitem__(self, index):
        video = self.train_video[index] if self.mode == 'train' else self.valid_video[index]
        label = self.train_label[index] if self.mode == 'train' else self.valid_label[index]
        return video, label

    def __len__(self):
        return len(self.train_video) if self.mode == 'train' else len(self.valid_video)


if __name__ == '__main__':
    a = TrimmedVideo()
    print(a.train_video.size())
    print(a.train_label)
