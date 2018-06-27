from utils import read_image, listdir
import torch
import random
import os


class Cifar100(object):
    def __init__(self, config, mode='test', transform=None):
        super(Cifar100, self).__init__()
        assert(mode == 'train' or mode == 'valid' or mode == 'eval')
        self.idx = 0
        self.config = config
        self.mode = mode
        self.transform = transform

        self.train_image, self.test_image = {}, {}
        self.base_label, self.novel_label = [], []
        self.load_file()

    def load_file(self):

        """
            load file for support set of each base/novel classes
        """
        if self.mode == 'train':
            base = self.config['base_dir']

            # base
            classes_dir = listdir(base, key=lambda x: x[-2:])
            classes_dir = [os.path.join(base, files) for files in classes_dir if files.startswith('class')]

            for directory in classes_dir:
                path = os.path.join(directory, 'train')
                image = read_image(path)
                _, label = os.path.split(directory)
                self.train_image[int(label[-2:])] = image
                self.base_label.append(int(label[-2:]))

        elif self.mode == 'valid':
            raise NotImplementedError

        else:
            novel = self.config['novel_dir']

            # novel
            classes_dir = listdir(novel, key=lambda x: x[-2:])
            classes_dir = [os.path.join(novel, files) for files in classes_dir if files.startswith('class')]

            for directory in classes_dir:
                path = os.path.join(directory, 'train')
                # TODO: random sample, currently take first 5 images from novel class
                image = read_image(path)[:5]
                _, label = os.path.split(directory)
                self.train_image[int(label[-2:])] = image
                self.novel_label.append(int(label[-2:]))

            print(self.novel_label)
            # test
            test = self.config['test_dir']
            self.test_set = read_image(test)

    def sample(self, data, sample_num):
        sample = random.sample(data, sample_num)
        return sample

    def __next__(self):
        if self.idx < self.__len__():
            self.idx += 1
            k_shot, n_query = self.config['train']['sample']['k_shot'], self.config['train']['sample']['n_query']
            n_way = self.config['train']['sample']['n_way']

            if self.mode == 'train':
                sample_classes = self.sample(self.base_label, n_way)
                sample_images, query_images = [], []
                for classes in sample_classes:
                    images = self.train_image[classes]
                    samples = self.sample(images, k_shot+n_query)
                    if self.transform is not None:
                        samples = [self.transform(item) for item in samples]

                    sample_images.extend(samples[:k_shot])
                    query_images.extend(samples[k_shot:])

                sample_images = torch.stack(sample_images)
                query_images = torch.stack(query_images)

                labels = [classes for classes in range(n_way) for _ in range(k_shot)]
                labels = torch.LongTensor(labels)

                return sample_images, query_images, labels

            elif self.mode == 'valid':
                raise NotImplementedError

            else:
                sample_images = []
                for classes in self.novel_label:
                    images = self.train_image[classes]
                    if self.transform is not None:
                        images = [self.transform(item) for item in images]
                    sample_images.extend(images)

                sample_images = torch.stack(sample_images)
                query_images = self.test_set[(self.idx-1)*k_shot*n_way:(self.idx*k_shot*n_way)]
                if self.transform is not None:
                    query_images = [self.transform(item) for item in query_images]
                query_images = torch.stack(query_images)

                return sample_images, query_images

        else:
            raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        if self.mode == 'train':
            return self.config['train']['episode']
        else:
            return len(self.test_set) // (self.config['train']['sample']['n_way'] *
                                          self.config['train']['sample']['k_shot'])



if __name__ == '__main__':
    import json
    configs = json.load(open('../configs/relationnet_config.json'))
    test = Cifar100(configs)
    print(test)
