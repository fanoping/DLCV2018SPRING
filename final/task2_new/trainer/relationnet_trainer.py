from datasets.dataset import Cifar100
from model.relationnet import Relationnet
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import torchvision.transforms as transforms
from torch.autograd import Variable
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
        # train
        self.dataset = Cifar100(config=self.config,
                                mode='train',
                                transform=transforms.Compose([
                                    transforms.ToTensor()
                                ]))

    def __build_model(self):
        self.model = Relationnet(self.config).cuda() if self.with_cuda else Relationnet(self.config)
        self.criterion = MSELoss().cuda() if self.with_cuda else MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.config['optimizer']['lr'])
        self.scheduler = StepLR(optimizer=self.optimizer,
                                gamma=self.config['optimizer']['scheduler']['gamma'],
                                step_size=self.config['optimizer']['scheduler']['step_size'])

    def train(self):
        self.model.train()
        total_loss, total_acc = 0, 0
        for episode, (support_image, query_image, label) in enumerate(self.dataset):
            self.scheduler.step(epoch=episode)
            support_image = Variable(support_image).cuda() if self.with_cuda else Variable(support_image)
            query_image = Variable(query_image).cuda() if self.with_cuda else Variable(query_image)
            label = Variable(label)

            self.model.zero_grad()
            output = self.model(support_image, query_image)
            one_hot_labels = torch.zeros(
                self.config['train']['sample']['n_query'] * self.config['train']['sample']['n_way'],
                self.config['train']['sample']['n_way']
            )
            one_hot_labels = one_hot_labels.scatter_(1, label.view(-1, 1), 1)
            one_hot_labels = one_hot_labels.cuda() if self.with_cuda else one_hot_labels

            loss = self.criterion(output, one_hot_labels)
            loss.backward()
            self.optimizer.step()

            _, result = torch.max(output, dim=1)
            acc = torch.eq(result.cpu(), label).float().mean()

            total_loss += loss.data.item()
            total_acc += acc
            print('[Episode: {}/{} ({:.3f}%)] Loss: {:.6f} Acc: {:.3f}'.format(
                episode + 1,
                len(self.dataset),
                100.0 * (episode + 1) / len(self.dataset),
                loss.data.item(),
                acc
            ), end='\r')
            sys.stdout.write('\033[K')

            if (episode + 1) % self.config['log_step'] == 0:
                print('[Episode: {}/{} ({:.3f}%)] Loss: {:.6f} Acc: {:.3f}'.format(
                    episode + 1,
                    len(self.dataset),
                    100.0 * (episode + 1) / len(self.dataset),
                    loss.data.item(),
                    acc
                ))

            self.loss_list.append(loss)
            self.acc_list.append(acc)

            assert (self.config['metric'] == 'accuracy' or self.config['metric'] == 'loss')
            metric = acc if self.config['metric'] == 'accuracy' else loss
            self.__save_checkpoint(episode + 1, metric)


        ave_loss = total_loss / len(self.dataset)
        ave_acc = total_acc / len(self.dataset)

        print("Loss: {:.6f} Acc: {:.4f}".format(ave_loss, ave_acc))

    def __save_checkpoint(self, episode, metric):
        state = {
            'structure': self.config['structure'],
            'episode': episode,
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

        filename = os.path.join(filepath, "epoch{}_checkpoint.pth.tar".format(episode))
        if episode % self.config['save']['save_freq'] == 0:
            torch.save(state, f=filename)

        best_filename = os.path.join(filepath, "best_checkpoint.pth.tar")
        if self.config['metric'] == 'accuracy':
            if self.max_acc < metric:
                torch.save(state, f=best_filename)
                print("Saving Epoch: {}, Updating acc {:.4f} to {:.4f}".format(episode, self.max_acc, metric))
                self.max_acc = metric
        else:
            if self.min_loss > metric:
                torch.save(state, f=best_filename)
                print("Saving Epoch: {}, Updating loss {:.6f} to {:.6f}".format(episode, self.min_loss, metric))
                self.min_loss = metric
