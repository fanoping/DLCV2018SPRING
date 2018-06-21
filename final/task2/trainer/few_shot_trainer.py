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
        # train
        self.train_dataset = Cifar100(config=self.config,
                                      mode='train',
                                      transform=transforms.Compose([
                                          transforms.ToTensor()
                                      ]))
        self.train_sampler = Sampler(labels=self.train_dataset.label,
                                     n_way=self.config['sampler']['train']['n_way'],
                                     k_shot=self.config['sampler']['train']['k_shot'],
                                     k_query=self.config['sampler']['train']['k_query'],
                                     n_episodes=self.config['sampler']['train']['episodes'])
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_sampler=self.train_sampler)

        # valid
        self.valid_dataset = Cifar100(config=self.config,
                                      mode='valid',
                                      transform=transforms.Compose([
                                          transforms.ToTensor()
                                      ]))
        self.valid_sampler = Sampler(labels=self.valid_dataset.label,
                                     n_way=self.config['sampler']['valid']['n_way'],
                                     k_shot=self.config['sampler']['valid']['k_shot'],
                                     k_query=self.config['sampler']['valid']['k_query'],
                                     n_episodes=self.config['sampler']['valid']['episodes'])
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
            total_loss, total_acc = 0, 0
            for episode, (image, label) in enumerate(self.train_dataloader):
                image = Variable(image).cuda() if self.with_cuda else Variable(image)
                label = Variable(label).cuda() if self.with_cuda else Variable(label)
                print(image.size())

                output = self.model(image)

                self.optimizer.zero_grad()
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.data[0]

                print('Epoch: {}/{} [Episode: {}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                        epoch,
                        self.config['epochs'],
                        episode,
                        len(self.train_dataloader),
                        100.0 * episode / len(self.train_dataloader),
                        loss.data[0]
                ), end='\r')
                sys.stdout.write('\033[K')

            ave_loss = total_loss / len(self.train_dataloader)
            print("Epoch: {}/{} Loss: {:.6f} Acc: {:.6f}".format(epoch,
                                                                 self.config['epochs'],
                                                                 total_loss / len(self.train_dataloader),
                                                                 total_acc / len(self.train_dataloader)))

    def __save_checkpoint(self):
        pass
