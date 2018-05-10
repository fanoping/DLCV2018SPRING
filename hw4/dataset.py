import argparse
import torch
from torch.utils.data import Dataset
from utils.preprocess import readfigs
import pandas as pd
import numpy as np


class CelebADataset(Dataset):
    def __init__(self, train_filepath, train_csvfile, test_filepath, test_csvfile, mode='train', transform=None):
        """
            train_data:
                description: training image
                type: ndarray
                shape: 40000 x image size (64 x 64 x 3)

            test_data:
                description: testing image
                type: ndarray
                shape: 2621 x image size (64 x 64 x 3)

            train_attr:
                attribute from train_csvfile
                size 40000 x 14 (1 img name + 13 attribute)

            test_attr:
                attribute from test_csvfile
                size 2621 x 14 (1 img name + 13 attribute)

            transform:
                mapping fig 0 ~ 255 to -1 ~ 1
                torchvision.transforms
                [toTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)]

        """
        self.train_data = readfigs(train_filepath)
        self.test_data = readfigs(test_filepath)
        self.train_attr = pd.read_csv(train_csvfile)
        self.test_attr = pd.read_csv(test_csvfile)
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index):
        if self.mode == 'train':
            data = self.train_data[index]
            label = self.train_attr.ix[index, 1:].as_matrix().astype('float')
            label = torch.FloatTensor(label)

        else:
            data = self.test_data[index]
            label = self.test_attr.ix[index, 1:].as_matrix().astype('float')
            label = torch.FloatTensor(label)

        if self.transform is not None:
            data = self.transform(data)

        return data, label

    def __len__(self):
        return len(self.train_data)


class GANDataset(CelebADataset):
    def __init__(self, train_filepath, train_csvfile, test_filepath, test_csvfile, transform=None):
        super(GANDataset, self).__init__(train_filepath, train_csvfile, test_filepath, test_csvfile)
        """
            train_data:
                description: training image
                type: ndarray
                shape: (40000 + 2621) x image size (64 x 64 x 3)

            train_attr:
                attribute from train_csvfile
                size (40000 + 2621) x 14 (1 img name + 13 attribute)

            transform:
                mapping fig 0 ~ 255 to -1 ~ 1
                torchvision.transforms
                [toTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)]
                output:
                    shape: 3 x 64 x 64 with all items between [-1, 1]
        """

        self.train_data = np.concatenate((self.train_data, self.test_data), axis=0)
        self.train_attr = pd.concat((self.train_attr, self.test_attr), ignore_index=True)
        self.transform = transform


    def __getitem__(self, index):
        data = self.train_data[index]
        label = self.train_attr.ix[index, 1:].as_matrix().astype('float')
        label = torch.FloatTensor(label)

        if self.transform is not None:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.train_data)


if __name__ == '__main__':
    import torchvision.transforms as transforms

    train_data_path = 'hw4_data/train'
    train_data_csv = 'hw4_data/train.csv'
    test_data_path = 'hw4_data/test'
    test_data_csv = 'hw4_data/test.csv'
    train_dataset = GANDataset(train_data_path,
                                 train_data_csv,
                                 test_data_path,
                                 test_data_csv,
                                 transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                 ]))

    print(train_dataset.train_data.shape)
