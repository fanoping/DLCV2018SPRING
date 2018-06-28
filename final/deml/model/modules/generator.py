from torchvision.models import resnet50
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        model = resnet50(pretrained=False)
        self.generator = nn.Sequential(*list(model.children())[:-1])

    def forward(self, image):
        concept = self.generator(image)
        return concept
