from .modules import Resnet50, Classfier, Vgg19
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.cnn = eval(args.pretrained.title())()
        if args.pretrained.upper() == 'VGG19':
            self.fc = Classfier(4096)
        elif args.pretrained.upper() == 'RESNET50':
            self.fc = Classfier(2048)
        print('Using {} pre-trained model.'.format(args.pretrained))

        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, frame):
        cnn_feature = self.cnn(frame)
        category = self.fc(cnn_feature.squeeze())
        return category
