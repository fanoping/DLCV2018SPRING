from torch.utils.data import Dataset
from scipy.misc import imread
import random
import os
# TODO: novel set random sample 1/5/10 images


class Cifar100(Dataset):
    def __init__(self, config, file='base', mode='support', transform=None):
        super(Cifar100, self).__init__()
        assert(file == 'base' or file == 'novel')
        assert(mode == 'support' or mode == 'query')
        self.config = config
        self.file = file
        self.mode = mode
        self.transform = transform

        self.image, self.label = [], []
        self.load_file()

    def load_file(self):
        if self.mode == 'support':
            """
                load file for support set of each base/novel classes
            """
            base = os.path.join(self.config['train_dir'], self.file)
            classes_dir = [os.path.join(base, files)
                           for files in sorted(os.listdir(base)) if files.startswith('class')]
            for idx, directory in enumerate(classes_dir):
                path = os.path.join(directory, 'train')
                image = [imread(os.path.join(path, image))
                         for image in sorted(os.listdir(path)) if image.endswith(".png")]
                label = [int(directory[-2:]) for _ in range(len(image))]
                self.image.extend(image)
                self.label.extend(label)
        else:
            """
                load file for query set of each base/novel classes
            """
            if self.file == 'base':
                base = os.path.join(self.config['train_dir'], 'base')
                classes_dir = [os.path.join(base, files)
                               for files in sorted(os.listdir(base)) if files.startswith('class')]
                for idx, directory in enumerate(classes_dir):
                    path = os.path.join(directory, 'test')
                    image = [imread(os.path.join(path, image))
                             for image in sorted(os.listdir(path)) if image.endswith(".png")]
                    label = [int(directory[-2:]) for _ in range(len(image))]
                    self.image.extend(image)
                    self.label.extend(label)
            else:
                self.image = [imread(os.path.join(self.config['test_dir'], image))
                              for image in sorted(os.listdir(self.config['test_dir']))]

    def __getitem__(self, index):
        image = self.transform(self.image[index]) if self.transform is not None else self.image[index]
        if self.file == 'base':
            label = self.label[index]
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.image)
