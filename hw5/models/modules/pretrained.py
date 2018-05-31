import torch
import torch.nn as nn
import torchvision.models as models


class Vgg19(nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        model = models.vgg19(pretrained=True)
        self.features = model.features
        self.classifier = nn.Sequential(*list(model.classifier.children())[:1])

    def forward(self, x):
        """
            output shape: batch size x 4096 x 1 x 1
        """
        hidden = self.features(x)
        output = self.classifier(hidden.view(hidden.size(0), -1))
        return output


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        model = models.resnet50(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        """
            output shape: batch size x 2048 x 1 x 1
        """
        output = self.feature(x)
        return output


class Densenet121(nn.Module):
    def __init__(self):
        super(Densenet121, self).__init__()
        model = models.densenet121(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])

    def forward(self, image):
        """
            output shape: batch size x 1024 x 7 x 7
        """
        output = self.feature(image)
        return output
