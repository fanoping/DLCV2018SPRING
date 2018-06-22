import numpy as np


class Sampler(object):
    def __init__(self, labels, n_way, k_samples, n_episodes):
        self.labels = labels
        self.n_way = n_way
        self.k_samples = k_samples
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for _ in range(self.n_episodes):
            all_idx = []
            classes, counts = np.unique(self.labels, return_counts=True)
            permutation = np.random.permutation(classes.shape[0])[:self.n_way]
            sample_classes, sample_counts = classes[permutation], counts[permutation]

            for sample_class, count in zip(sample_classes, sample_counts):
                class_idx = np.random.choice(count, self.k_samples, replace=False)
                start_idx = self.labels.index(sample_class)
                class_idx += start_idx
                all_idx.extend(class_idx)

            yield all_idx
