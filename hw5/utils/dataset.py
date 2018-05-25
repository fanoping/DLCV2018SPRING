from utils.reader import getVideoList
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle


class TrimmedVideo(Dataset):
    def __init__(self, mode='train', pool=True):
        super(TrimmedVideo, self).__init__()
        self.with_cuda = True
        self.mode = mode
        self.pool = pool
        self.__load_data()

    def __load_data(self):
        print("Loading {} data".format(self.mode))
        if self.mode == 'train':
            self.train_video = torch.load('train_video.tar')

            if self.pool:
                self.train_video = [torch.mean(frames, dim=0) for _, frames in self.train_video.items()]
                self.train_video = torch.stack(self.train_video)
            print(self.train_video.size())
            self.train_label = getVideoList('HW5_data/TrimmedVideos/label/gt_train.csv')['Action_labels']
            self.train_label = np.array(self.train_label).astype(np.uint8)
            self.train_label = torch.LongTensor(self.train_label)

        else:
            self.valid_video = torch.load('valid_video.tar')

            if self.pool:
                self.valid_video = [torch.mean(frames, dim=0) for _, frames in self.valid_video.items()]
                self.valid_video = torch.stack(self.valid_video)

            self.valid_label = getVideoList('HW5_data/TrimmedVideos/label/gt_valid.csv')['Action_labels']
            self.valid_label = np.array(self.valid_label).astype(np.uint8)
            self.valid_label = torch.LongTensor(self.valid_label)

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
