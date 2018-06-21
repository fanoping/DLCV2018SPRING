from torch.utils.data import Dataset
from scipy.misc import imread
import numpy as np
import os


class Cifar100(Dataset):
    def __init__(self, config, mode='train', transform=None):
        super(Cifar100, self).__init__()
        assert(mode=='train' or mode=='valid' or mode=='test')
        self.config = config
        self.mode = mode
        self.transform = transform

        self.image = []
        self.label = []

        self.load_file()

    def load_file(self):
        if self.mode == 'train':
            # load file from "base/train" and "novel/train" of each base/novel classes
            types = ['base', 'novel']
            for class_type in types:
                base = os.path.join(self.config['train_dir'], class_type)
                classes_dir = [os.path.join(base, files) for files in sorted(os.listdir(base)) if files.startswith('class')]
                for idx, directory in enumerate(classes_dir):
                    train_path = os.path.join(directory, self.mode)
                    image = [imread(os.path.join(train_path, image))
                             for image in sorted(os.listdir(train_path)) if image.endswith(".png")]
                    label = [int(directory[-2:]) for _ in range(len(image))]

                    self.image.extend(image)
                    self.label.extend(label)

        elif self.mode == 'valid':
            # load file from "base/test" of each classes
            base = os.path.join(self.config['train_dir'], 'base')
            classes_dir = [os.path.join(base, files) for files in sorted(os.listdir(base)) if files.startswith('class')]
            for idx, directory in enumerate(classes_dir):
                train_path = os.path.join(directory, 'test')
                image = [imread(os.path.join(train_path, image))
                         for image in sorted(os.listdir(train_path)) if image.endswith(".png")]
                label = [int(directory[-2:]) for _ in range(len(image))]

                self.image.extend(image)
                self.label.extend(label)

        else:
            self.image = [imread(os.path.join(self.config['test_dir'], image))
                          for image in sorted(os.listdir(self.config['test_dir']))]

    def __getitem__(self, index):
        image = self.transform(self.image[index]) if self.transform is not None else self.image[index]
        label = self.label[index]

        return image if self.mode == 'test' else image, label

    def __len__(self):
        return len(self.image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', default='../datasets/task2-dataset')
    parser.add_argument('--test-dir', default='../datasets/test')
    import torchvision.transforms as transforms

    transform = transforms.Compose([transforms.ToTensor()])
    data = Cifar100(parser.parse_args(), transform=transform)
