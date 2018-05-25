import torch
import torch.nn as nn
import torchvision.models as models


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        model = models.vgg19(pretrained=True).cuda()
        self.features = model.features
        self.classifier = nn.Sequential(*list(model.classifier.children())[:1])

    def forward(self, x):
        hidden = self.features(x)
        output = self.classifier(hidden.view(hidden.size(0), -1))
        return output


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        model = models.resnet50(pretrained=True).cuda()
        self.feature = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        """
            output shape: batch size x 2048 x 1 x 1
        """
        output = self.feature(x)
        return output
