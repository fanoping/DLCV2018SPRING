import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
# TODO: loss function implementation


class ProtoLoss(nn.Module):
    def __init__(self, config):
        super(ProtoLoss, self).__init__()
        self.config = config

    def forward(self, support_image, support_label, query_image, query_label):
        """
        calculating euclidean distance for query image
        :param support_image: (n_way x k_shot) x 64
        :param support_label: (n_way x k_shot),
        :param query_image: (n_way x k_shot) x 64
        :param query_label: (n_way x k_shot),
        :return:
        """

        return self.__protoloss(support_image, support_label, query_image, query_label)

    def __protoloss(self, support_image, support_label, query_image, query_label):
        support_label_cpu = support_label.cpu().numpy() if self.config['cuda'] else support_label.numpy()
        classes = np.unique(support_label_cpu)

        n_class = len(classes)
        n_query = query_image.size(0)

        prototype = []
        for single_class in classes:
            class_images = [support_image[i] for i, label in enumerate(support_label_cpu) if label == single_class]
            class_images = torch.stack(class_images)
            class_prototype = class_images.mean(dim=0).squeeze(0)
            prototype.append(class_prototype)

        prototype = torch.stack(prototype).cuda() if self.config['cuda'] else torch.stack(prototype)
        distance = self.euclidean_dist(query_image, prototype)
        log_p_y = F.log_softmax(-distance, dim=1)

        query_label_cpu = query_label.cpu().numpy() if self.config['cuda'] else query_label.numpy()
        classes = np.unique(query_label_cpu)

        # FIXME: still not correct
        class_dict = dict(zip(classes, [i for i in range(len(classes))]))
        temp_label = np.array([class_dict[idx] for idx in query_label_cpu])

        target_inds = torch.arange(0, n_class).view(n_class, 1).expand(n_class, n_query).long() #20*100
        target_inds = Variable(target_inds, requires_grad=False)
        target_inds = target_inds.cuda() if self.config['cuda'] else target_inds

        loss_val = -log_p_y.gather(0, target_inds).squeeze().view(-1).mean()

        _, result = log_p_y.max(dim=0)
        print(result.shape)
        acc_val = np.mean(result.cpu().data.numpy() == temp_label)

        return loss_val, acc_val

    def euclidean_dist(self, x, y):
        '''
        Compute euclidean distance between two tensors
        code from: https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/utils.py
        '''
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)
