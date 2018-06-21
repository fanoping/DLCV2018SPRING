import numpy as np
import torch


class Sampler(object):
    def __init__(self, labels, n_way, k_shot, k_query, iterations):
        super(Sampler, self).__init__()

        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.idxs = range(len(self.labels))

        self.label_tens = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.label_tens = torch.Tensor(self.label_tens)

        self.label_lens = torch.zeros(len(self.classes))
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.label_tens[label_idx, np.where(np.isnan(self.label_tens[label_idx]))[0][0]] = idx
            self.label_lens[label_idx] += 1

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        yield "A"

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations


if __name__ == '__main__':
    from datasets import Cifar100
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', default='../datasets/task2-dataset')
    parser.add_argument('--test-dir', default='../datasets/test')
    dataset = Cifar100(parser.parse_args(), mode='valid')

    test = Sampler(dataset.label, 10, 10, 1)

    a = iter(test)
    for i in a:
        print(i)
