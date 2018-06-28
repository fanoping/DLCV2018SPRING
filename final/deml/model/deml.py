from .modules import Discriminator, Generator, MetaLearning
import torch.nn as nn
import torch


class DEML(nn.Module):
    def __init__(self, config):
        super(DEML, self).__init__()
        self.config = config
        self.concept_generator = Generator()
        self.concept_discrimiator = Discriminator()
        self.meta_learner = MetaLearning()

    def forward(self, samples, query):
        # TODO: samples input configuration
        concept = self.concept_generator(samples)
        classes = self.concept_discrimiator(concept)
        meta_output = self.meta_learner(concept)

        return classes, meta_output
