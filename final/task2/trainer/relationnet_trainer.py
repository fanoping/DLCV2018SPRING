from utils.datasets import Cifar100
from utils.sampler import Sampler
from model.relationnet import Relationnet
from torch.nn import MSELoss
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
import torch
import json
import sys
import os


class RelationnetTrainer:
    def __init__(self, config):
        self.config = config
        self.with_cuda = config['cuda']

        self.__load()
        self.__build_model()

        self.loss_list, self.acc_list = [], []
        self.min_loss = float('inf')
        self.max_acc = 0

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
        self.model = Relationnet(self.config).cuda() if self.with_cuda else Relationnet(self.config)
        self.criterion = MSELoss().cuda() if self.with_cuda else MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.config['optimizer']['lr'])
        self.scheduler = StepLR(optimizer=self.optimizer,
                                gamma=self.config['optimizer']['scheduler']['gamma'],
                                step_size=self.config['optimizer']['scheduler']['step_size'])

    def train(self):
        for epoch in range(1, self.config['epochs'] + 1):
            self.model.train()
            self.scheduler.step(epoch=epoch)
            total_loss, total_acc = 0, 0
            for episode, ((support_image, support_label), (query_image, query_label)) in \
                    enumerate(zip(self.support_dataloader, self.query_dataloader)):
                support_image = Variable(support_image).cuda() if self.with_cuda else Variable(support_image)
                support_label = Variable(support_label).cuda() if self.with_cuda else Variable(support_label)
                query_image = Variable(query_image).cuda() if self.with_cuda else Variable(query_image)
                query_label = Variable(query_label).cuda() if self.with_cuda else Variable(query_label)

                self.model.zero_grad()
                output = self.model(support_image, query_image)
                one_hot_labels = Variable(torch.zeros(
                        self.config['sampler']['train']['k_query'] * self.config['sampler']['train']['n_way'],
                        self.config['sampler']['train']['n_way']
                ).scatter_(1, query_label.view(-1, 1), 1))
                one_hot_labels = one_hot_labels.cuda() if self.with_cuda else one_hot_labels

                loss = self.criterion(output, one_hot_labels)
                loss.backward()
                self.optimizer.step()

                _, result = torch.max(output, dim=1)
                acc = torch.eq(result.cpu(), query_label).float().mean()

                total_loss += loss.data.item()
                total_acc += acc

                print('Epoch: {}/{} [Episode: {}/{} ({:.0f}%)] Loss: {:.6f} Acc: {:.3f}'.format(
                        epoch,
                        self.config['epochs'],
                        episode + 1,
                        len(self.support_dataloader),
                        100.0 * (episode+1) / len(self.support_dataloader),
                        loss.data.item(),
                        acc
                ), end='\r')
                sys.stdout.write('\033[K')

            ave_loss = total_loss / len(self.support_dataloader)
            ave_acc = total_acc / len(self.support_dataloader)

            self.loss_list.append(ave_loss)
            self.acc_list.append(ave_acc)

            print("Epoch: {}/{} Loss: {:.6f} Acc: {:.4f}".format(epoch,
                                                                 self.config['epochs'],
                                                                 total_loss / len(self.support_dataloader),
                                                                 total_acc / len(self.support_dataloader)))

            assert(self.config['metric'] == 'accuracy' or self.config['metric'] == 'loss')
            metric = ave_acc if self.config['metric'] == 'accuracy' else ave_loss
            self.__save_checkpoint(epoch, metric)

    def __save_checkpoint(self, epoch, metric):
        state = {
            'structure': self.config['structure'],
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': self.loss_list,
            'accuracy': self.acc_list,
        }

        filepath = os.path.join("checkpoints", self.config['save']['dir'])
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        with open(os.path.join(filepath, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4, sort_keys=False)

        filename = os.path.join(filepath, "epoch{}_checkpoint.pth.tar".format(epoch))
        if epoch % self.config['save']['save_freq'] == 0:
            torch.save(state, f=filename)

        best_filename = os.path.join(filepath, "best_checkpoint.pth.tar")
        if self.config['metric'] == 'accuracy':
            if self.max_acc < metric:
                torch.save(state, f=best_filename)
                print("Saving Epoch: {}, Updating acc {:.4f} to {:.4f}".format(epoch, self.max_acc, metric))
                self.max_acc = metric
        else:
            if self.min_loss > metric:
                torch.save(state, f=best_filename)
                print("Saving Epoch: {}, Updating loss {:.6f} to {:.6f}".format(epoch, self.min_loss, metric))
                self.min_loss = metric
