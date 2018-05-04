import argparse
import torch
from torch.utils.data import Dataset
from utils.preprocess import readfigs


class CelebADataset(Dataset):
    def __init__(self, filepath):
        self.train_data = readfigs(filepath)
        self.train_data = torch.FloatTensor(self.train_data)

    def __getitem__(self, index):
        return self.train_data[index], self.train_data[index]

    def __len__(self):
        return len(self.train_data)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class CIFAR10DataLoader:
    def __init__(self, args):
        if args.arch == 'CIFAR10':
            # Data Loading


            transform_train = transforms.Compose([
                transforms.ToTensor()
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor()
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

            self.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle)

            test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
            self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

            self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        else:
            raise ValueError('The dataset should be CIFAR10')


if __name__ == '__main__':
    #A = CelebADataset('hw4_data/train')
    #print(A[3])

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=1)
    parser.add_argument('--arch', default='CIFAR10')
    parser.add_argument('--shuffle', default=False)
    parser.add_argument('--cuda', default=True)
    B = CIFAR10DataLoader(parser.parse_args())
    print(B.train_loader)
    #print(B.train_set)
    print()
