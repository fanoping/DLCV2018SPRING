from utils.datasets import Cifar100
from utils.sampler import Sampler
from model.protonet import Protonet
from model.loss import ProtoLoss
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
import sys



class FewshotTrainer:
    def __init__(self, config):
        self.config = config
        self.with_cuda = config['cuda']

        self.__load()
        self.__build_model()

        self.loss_list, self.acc_list = [], []

    def __load(self):
        # support
        self.support_dataset = Cifar100(config=self.config,
                                        file='base',
                                        mode='support',
                                        transform=transforms.Compose([
                                            transforms.ToTensor()
                                        ]))
        self.support_sampler = Sampler(labels=self.support_dataset.label,
                                       n_way=self.config['sampler']['train']['n_way'],
                                       k_samples=self.config['sampler']['train']['k_shot'],
                                       n_episodes=self.config['sampler']['train']['episodes'])
        self.support_dataloader = DataLoader(dataset=self.support_dataset,
                                             batch_sampler=self.support_sampler)
        # query
        self.query_dataset = Cifar100(config=self.config,
                                      file='base',
                                      mode='query',
                                      transform=transforms.Compose([
                                          transforms.ToTensor()
                                      ]))
        self.query_sampler = Sampler(labels=self.query_dataset.label,
                                     n_way=self.config['sampler']['train']['n_way'],
                                     k_samples=self.config['sampler']['train']['k_query'],
                                     n_episodes=self.config['sampler']['train']['episodes'])
        self.query_dataloader = DataLoader(dataset=self.query_dataset,
                                           batch_sampler=self.query_sampler)

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
            total_loss, total_acc = 0, 0
            for episode, ((support_image, support_label), (query_image, query_label)) in \
                         enumerate(zip(self.support_dataloader, self.query_dataloader)):
                support_image = Variable(support_image).cuda() if self.with_cuda else Variable(support_image)
                support_label = Variable(support_label).cuda() if self.with_cuda else Variable(support_label)
                query_image = Variable(query_image).cuda() if self.with_cuda else Variable(query_image)
                query_label = Variable(query_label).cuda() if self.with_cuda else Variable(query_label)

                support_output = self.model(support_image)
                query_image = self.model(query_image)

                self.optimizer.zero_grad()
                loss = self.criterion(support_output, support_label, query_image, query_label)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.data[0]

                print('Epoch: {}/{} [Episode: {}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                        epoch,
                        self.config['epochs'],
                        episode,
                        len(self.support_dataloader),
                        100.0 * episode / len(self.support_dataloader),
                        loss.data[0]
                ), end='\r')
                sys.stdout.write('\033[K')

            ave_loss = total_loss / len(self.support_dataloader)
            print("Epoch: {}/{} Loss: {:.6f} Acc: {:.6f}".format(epoch,
                                                                 self.config['epochs'],
                                                                 total_loss / len(self.support_dataloader),
                                                                 total_acc / len(self.support_dataloader)))

    def __save_checkpoint(self):
        pass
