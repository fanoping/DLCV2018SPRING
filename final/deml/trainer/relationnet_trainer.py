from datasets.dataset import Cifar100
from model.relationnet import Relationnet
from utils import mkdir
from torch.nn import MSELoss
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
import torch
import json
import csv
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
        mkdir(os.path.join('saved', self.config['save']['dir']))
        mkdir(os.path.join("checkpoints", self.config['save']['dir']))

    def __load(self):
        # train
        self.train_dataset = Cifar100(config=self.config,
                                      mode='train',
                                      transform=transforms.Compose([
                                          transforms.ToTensor()
                                      ]))
        # test
        self.test_dataset = Cifar100(config=self.config,
                                     mode='eval',
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
        total_loss, total_acc = 0, 0
        for episode, (support_image, query_image, label) in enumerate(self.train_dataset):
            self.model.train()
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
                len(self.train_dataset),
                100.0 * (episode + 1) / len(self.train_dataset),
                loss.data.item(),
                acc
            ), end='\r')
            sys.stdout.write('\033[K')

            if (episode + 1) % self.config['log_step'] == 0:
                print('[Episode: {}/{} ({:.3f}%)] Loss: {:.6f} Acc: {:.3f}'.format(
                    episode + 1,
                    len(self.train_dataset),
                    100.0 * (episode + 1) / len(self.train_dataset),
                    loss.data.item(),
                    acc
                ))

            self.loss_list.append(loss)
            self.acc_list.append(acc)

            assert (self.config['metric'] == 'accuracy' or self.config['metric'] == 'loss')
            metric = acc if self.config['metric'] == 'accuracy' else loss
            self.__save_checkpoint(episode + 1, metric)

            if (episode + 1) % self.config['eval_freq'] == 0:
                self.eval(episode+1)

        ave_loss = total_loss / len(self.train_dataset)
        ave_acc = total_acc / len(self.train_dataset)

        print("Loss: {:.6f} Acc: {:.4f}".format(ave_loss, ave_acc))

    def eval(self, episode=None):
        print("Evaluation...")
        with torch.no_grad():
            self.model.eval()
            label = self.test_dataset.novel_label
            results = []
            for _, (support_image, query_image) in enumerate(self.test_dataset):
                support_image = Variable(support_image).cuda() if self.with_cuda else Variable(support_image)
                query_image = Variable(query_image).cuda() if self.with_cuda else Variable(query_image)

                output = self.model(support_image, query_image)
                _, result = torch.max(output, dim=1)
                results.append(result)
            results = torch.cat(results, dim=0).data.cpu().numpy().tolist()
            results = [label[idx] for idx in results]

        filename = os.path.join('saved',
                                self.config['save']['dir'],
                                'test.csv' if episode is None else 'test_episode{}.csv'.format(episode))
        with open(filename, 'w') as f:
            s = csv.writer(f, delimiter=',', lineterminator='\n')
            s.writerow(["image_id", "predicted_label"])
            for idx, predict_label in enumerate(results):
                s.writerow([idx, predict_label])
        print("Saving inference label csv as {}".format(filename))

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
        
        with open(os.path.join(filepath, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4, sort_keys=False)

        filename = os.path.join(filepath, "episode{}_checkpoint.pth.tar".format(episode))
        if episode % self.config['save']['save_freq'] == 0:
            torch.save(state, f=filename)

        best_filename = os.path.join(filepath, "best_checkpoint.pth.tar")
        if self.config['metric'] == 'accuracy':
            if self.max_acc < metric:
                torch.save(state, f=best_filename)
                print("Saving Episode: {}, Updating acc {:.4f} to {:.4f}".format(episode, self.max_acc, metric))
                self.max_acc = metric
        else:
            if self.min_loss > metric:
                torch.save(state, f=best_filename)
                print("Saving Episode: {}, Updating loss {:.6f} to {:.6f}".format(episode, self.min_loss, metric))
                self.min_loss = metric
