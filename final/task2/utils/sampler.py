import numpy as np


class Sampler(object):
    def __init__(self, labels, n_way, k_shot, k_query, n_episodes):
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            idx = 0
            all_idx = []
            classes, counts = np.unique(self.labels, return_counts=True)
            permutation = np.random.permutation(classes.shape[0])[:self.n_way]
            classes, counts = classes[permutation], counts[permutation]

            for count in counts:
                class_idx = np.random.choice(count, self.k_shot + self.k_query, replace=False)
                class_idx += idx
                idx += count
                all_idx.extend(class_idx)
            yield all_idx
