from .modules import Classfier
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        if args.pretrained.title() == 'Vgg19':
            self.fc = Classfier(4096, mode='rnn')
        elif args.pretrained.title() == 'Resnet50':
            self.fc = Classfier(2048, mode='cnn')
        elif args.pretrained.title() == 'Densenet121':
            self.fc = Classfier(1024*7*7, mode='rnn')

    def forward(self, frame):
        category = self.fc(frame)
        return category
