import torch.nn as nn


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predict, target):
        """

        :param predict: classifier output, shape = (max_seq_len, batch_size, number of classes(11))
        :param target: target output, shape = (batch_size, max_seq_len)
        :return: loss
        """
        loss_fn = nn.CrossEntropyLoss()
        loss = 0
        batch_size = predict.size(0)

        for i in range(batch_size):
            partial_loss = loss_fn(predict[i], target[i])
            loss += partial_loss
        loss = loss / batch_size

        return loss
