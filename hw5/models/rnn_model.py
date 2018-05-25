from .modules import Resnet50, Classfier, Vgg19, Encoder
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()
        self.cnn = eval(args.pretrained.title())()
        self.encoder = Encoder()
        if args.pretrained.upper() == 'VGG19':
            self.fc = Classfier(4096)
        elif args.pretrained.upper() == 'RESNET50':
            self.fc = Classfier(2048)
        print('Using {} pre-trained model.'.format(args.pretrained))

        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, frames):
        cnn_feature = self.cnn(frames)
        enc_out, hidden_out = self.encoder(cnn_feature)
        category = self.fc(enc_out)
        return category
