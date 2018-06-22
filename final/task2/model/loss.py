import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import numpy as np
# TODO: loss function to be verified


class ProtoLoss(nn.Module):
    def __init__(self, config):
        super(ProtoLoss, self).__init__()
        self.config = config

    def forward(self, support_image, support_label, query_image, query_label):
        """
        calculating euclidean distance for query image
        Reference: https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
        :param support_image: (n_way x k_shot) x 64
        :param support_label: (n_way x k_shot),
        :param query_image: (n_way x k_query) x 64
        :param query_label: (n_way x k_query),
        :return:
        """
        return self.__protoloss(support_image, support_label, query_image, query_label)

    def __protoloss(self, support_image, support_label, query_image, query_label):
        support_label_cpu = support_label.cpu().numpy() if self.config['cuda'] else support_label.numpy()
        classes = np.unique(support_label_cpu)

        n_class = len(classes)
        n_query = self.config['sampler']['train']['k_query']

        prototype = []
        for single_class in classes:
            class_images = [support_image[i] for i, label in enumerate(support_label_cpu) if label == single_class]
            class_images = torch.stack(class_images)
            class_prototype = class_images.mean(dim=0).squeeze(0)
            prototype.append(class_prototype)
        prototype = torch.stack(prototype).cuda() if self.config['cuda'] else torch.stack(prototype)

        distance = self.__euclidean_dist(query_image, prototype)
        log_p_y = f.log_softmax(-distance, dim=1).view(n_class, n_query, -1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        target_inds = target_inds.cuda() if self.config['cuda'] else target_inds

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, result = log_p_y.max(dim=2)
        acc_val = torch.eq(result, target_inds.squeeze()).float().mean()

        return loss_val, acc_val

    def __euclidean_dist(self, x, y):
        """
        Compute euclidean distance between two tensors
        Reference: https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/utils.py
        """
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)
