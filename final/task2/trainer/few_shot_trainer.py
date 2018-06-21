from utils.datasets import Cifar100
from utils.sampler import Sampler
from model.protonet import Protonet
from model.loss import ProtoLoss
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam


class FewshotTrainer:
    def __init__(self, config):
        self.config = config
        self.with_cuda = config['cuda']

        self.__load()
        self.__build_model()

    def __load(self):
        # train
        self.train_dataset = Cifar100(config=self.config,
                                      mode='train',
                                      transform=transforms.Compose([
                                          transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                                                               std=(0.2675, 0.2565, 0.2761))
                                      ]))
        self.train_sampler = Sampler(labels=self.train_dataset.label,
                                     n_way=self.config['sampler']['train']['n_way'],
                                     k_shot=self.config['sampler']['train']['k_shot'],
                                     k_query=self.config['sampler']['train']['k_query'],
                                     iterations=self.config['sampler']['iterations'])
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_sampler=self.train_sampler)

        # valid
        self.valid_dataset = Cifar100(config=self.config,
                                      mode='valid',
                                      transform=transforms.Compose([
                                          transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                                                               std=(0.2675, 0.2565, 0.2761))
                                      ]))
        self.valid_sampler = Sampler(labels=self.valid_dataset.label,
                                     n_way=self.config['sampler']['valid']['n_way'],
                                     k_shot=self.config['sampler']['valid']['k_shot'],
                                     k_query=self.config['sampler']['valid']['k_query'],
                                     iterations=self.config['sampler']['iterations'])
        self.valid_dataloader = DataLoader(dataset=self.valid_dataset,
                                           batch_sampler=self.valid_sampler)

    def __build_model(self):
        self.model = Protonet(self.config).cuda() if self.with_cuda else Protonet(self.config)
        self.criterion = ProtoLoss().cuda() if self.with_cuda else ProtoLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.config['optimizer']['lr'])
        self.scheduler = StepLR(optimizer=self.optimizer,
                                gamma=self.config['optimizer']['scheduler']['gamma'],
                                step_size=self.config['optimizer']['scheduler']['step_size'])

    def train(self):
        for epoch in range(self.config['epochs']):
            self.model.train()
            self.scheduler.step(epoch=epoch)
            train_iter = iter(self.train_dataloader)
            for a in train_iter:
                print(a)

    def __save_checkpoint(self):
        pass
