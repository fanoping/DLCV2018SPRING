from model.modules import Embedder, Relation
import torch.nn as nn
import torch


class Relationnet(nn.Module):
    def __init__(self, config):
        super(Relationnet, self).__init__()
        self.config = config
        self.embedder = Embedder(config['model']['feature_size'], config['model']['hidden_size'])
        self.relation = Relation(config['model']['feature_size'],
                                 config['model']['hidden_size'],
                                 config['model']['relation_dim'])

    def forward(self, samples, batches):
        sample_features = self.embedder(samples)
        sample_features = sample_features.view(self.config['sampler']['train']['n_way'],
                                               self.config['sampler']['train']['k_shot'],
                                               self.config['model']['feature_size'], 6, 6)

        # element-wise sum if k-shot, k > 1
        sample_features = torch.sum(sample_features, dim=1).squeeze(1)
        sample_features_ext = sample_features.unsqueeze(0).repeat(self.config['sampler']['train']['k_query'] *
                                                                  self.config['sampler']['train']['n_way'],
                                                                  1, 1, 1, 1)
        batch_features = self.embedder(batches)
        batch_features_ext = batch_features.unsqueeze(0).repeat(self.config['sampler']['train']['n_way'], 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)

        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2)
        relation_pairs = relation_pairs.view(-1, self.config['model']['feature_size']*2, 6, 6)
        relations = self.relation(relation_pairs).view(-1, self.config['sampler']['train']['n_way'])

        return relations


if __name__ == '__main__':
    import json
    configs = json.load(open('../configs/relationnet_config.json'))
    test = Relationnet(configs)
    import torch
    from torch.autograd import Variable

    a = Variable(torch.ones((100, 3, 32, 32)))
    b = Variable(torch.ones((100, 3, 32, 32)))
    out = test(a, b)
