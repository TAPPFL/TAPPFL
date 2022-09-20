#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy

import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from utils import private_label_obtention, get_data, compute_metrics, jsd_loss
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
from sklearn.preprocessing import StandardScaler

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, data, labels, private_labels, idxs=None):
        self.data = np.stack([data[col].values for col in data.columns.values], 1)
        self.data = torch.tensor(self.data, dtype=torch.float)
        self.labels = torch.Tensor(labels).long()
        self.private_labels = torch.Tensor(private_labels).long()
        self.idxs = idxs
        if self.idxs is not None:
            self.idxs = [int(i) for i in idxs]

    def __len__(self):
        if self.idxs is not None:
            return len(self.idxs)
        else:
            return len(self.data)

    def __getitem__(self, item):
        if self.idxs is not None:
            data = self.data[self.idxs[item]]
            labels = self.labels[self.idxs[item]]
            private_labels = self.private_labels[self.idxs[item]]
            return data, labels, private_labels
        else:
            data = self.data[item]
            labels = self.labels[item]
            private_labels = self.private_labels[item]
            return data, labels, private_labels


class Pretrain(object):
    def __init__(self, args, dataset, logger):
        self.args = args
        self.logger = logger
        data_aux, labels_aux, private_labels_aux = get_data(args, dataset)
        self.trainloader = DataLoader(DatasetSplit(data_aux, labels_aux, private_labels_aux), batch_size=128, shuffle=False)
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def pretrain_fe(self, fe_model, classifier):
        fe_model.train()
        classifier.train()

        if self.args.optimizer == 'sgd':
            optimizer_fe = torch.optim.SGD(fe_model.parameters(), self.args.lr_fe, 0.5)
            optimizer_classifier = torch.optim.SGD(classifier.parameters(), self.args.lr_classifier, 0.5) #Should be model.privacy (create a privacy model)

        elif self.args.optimizer == 'adam':
            optimizer_fe = torch.optim.Adam(fe_model.parameters(), self.args.lr_fe, weight_decay=1e-4)
            optimizer_classifier = torch.optim.Adam(classifier.parameters(), self.args.lr_classifier, weight_decay=1e-4)

        for iter in range(self.args.pretrain_ep):
            for batch_idx, (data, labels, private_labels) in enumerate(self.trainloader):
                data, labels, private_labels = data.to(self.device), labels.to(self.device), private_labels.to(self.device)

                # The feature extractor is trained to accurately predict the real label
                feature = fe_model(data)
                pred = classifier(feature)
                loss = self.criterion(pred, labels)

                optimizer_fe.zero_grad()
                optimizer_classifier.zero_grad()
                loss.backward()
                optimizer_fe.step()
                optimizer_classifier.step()

        return fe_model, classifier

    def eval_pretrain_fe(self, fe_model, classifier):
        
        fe_model.eval()
        classifier.eval()

        for batch_idx, (data, labels, private_labels) in enumerate(self.trainloader):
            list_acc, list_loss = [], []
            data, labels, private_labels = data.to(self.device), labels.to(self.device), private_labels.to(self.device)

            feature = fe_model(data)
            pred = classifier(feature)
            loss = self.criterion(pred, labels)
            acc = float(compute_metrics(pred, labels))
            list_loss.append(loss.item())
            list_acc.append(acc)

        print('Average loss on pretraining: {} | Average accuracy on pretraining: {}%'.format(round(sum(list_loss)/len(list_loss),3),round(sum(list_acc)*100/len(list_acc),3)))

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.testloader, self.plotloader = self.train_val_test(
            args, dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, args, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_test = idxs[int(0.2*len(idxs)):]

        data_aux, labels_aux, private_labels_aux = get_data(args, dataset)

        plotloader = DataLoader(DatasetSplit(data_aux, labels_aux, private_labels_aux), batch_size=128, shuffle=False)

        n=10
        if args.dataset == 'credit':
            n=2

        trainloader = DataLoader(DatasetSplit(data_aux, labels_aux, private_labels_aux, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        #validloader = DataLoader(DatasetSplit(data_aux, labels_aux, private_labels_aux, idxs_val),
                                 #batch_size=int(len(idxs_val)/2), shuffle=False)
        testloader = DataLoader(DatasetSplit(data_aux, labels_aux, private_labels_aux, idxs_test),
                                batch_size=int(len(idxs_test)/n), shuffle=False)
        return trainloader, testloader, plotloader

    def train_fe(self, fe_model, classifier, mi_model, global_round):
        # Set mode to train models
        fe_model.train()
        classifier.train()
        mi_model.train()

        if self.args.optimizer == 'sgd':
            optimizer_fe = torch.optim.SGD(fe_model.parameters(), self.args.lr_fe, 0.5)
            optimizer_classifier = torch.optim.SGD(classifier.parameters(), self.args.lr_classifier, 0.5) #Should be model.privacy (create a privacy model)
            optimizer_mi = torch.optim.SGD(mi_model.parameters(), self.args.lr_mi_estimator, 0.5) #Should be a model.utility (create an utility model)

        elif self.args.optimizer == 'adam':
            optimizer_fe = torch.optim.Adam(fe_model.parameters(), self.args.lr_fe, weight_decay=1e-4)
            optimizer_classifier = torch.optim.Adam(classifier.parameters(), self.args.lr_classifier, weight_decay=1e-4)
            optimizer_mi = torch.optim.Adam(mi_model.parameters(), self.args.lr_mi_estimator, weight_decay=1e-4)


        for iter in range(self.args.local_ep):
            for batch_idx, (data, labels, private_labels) in enumerate(self.trainloader):
                data, labels, private_labels = data.to(self.device), labels.to(self.device), private_labels.to(self.device)

                # Necessary for JSD loss. x' is an random input data sampled independently of the same distribution of x
                aux1 = data[1:].clone()
                aux2 = data[0].unsqueeze(0).clone()
                x_prime = torch.cat((aux1, aux2), dim=0)

                optimizer_fe.zero_grad()
                optimizer_classifier.zero_grad()
                optimizer_mi.zero_grad()

                feature = fe_model(data)
                pred = classifier(feature)
                private_labels = private_labels.to(self.device)
                loss_classifier = self.criterion(pred, labels)
                mi_value = -jsd_loss(mi_model, data, feature, private_labels, x_prime)
                total_loss = -self.args.tradeoff_lambda*loss_classifier + (1-self.args.tradeoff_lambda)*mi_value

                total_loss.backward(retain_graph=True)
                loss_classifier.backward(retain_graph=True)
                mi_value.backward()

                optimizer_fe.step()
                optimizer_classifier.step()
                optimizer_mi.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tTotal loss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(data),
                        len(self.trainloader.dataset),
                                            100. * batch_idx / len(self.trainloader), total_loss.item()))
                self.logger.add_scalar('loss', total_loss.item())

        return fe_model.state_dict()

    def eval_train(self, fe_model, classifier, mi_model):
        fe_model.eval()
        classifier.eval()
        mi_model.eval()

        list_acc_utility, list_acc_privacy, list_loss_utility, list_loss_privacy = [], [], [], []

        for batch_idx, (data, labels, private_labels) in enumerate(self.trainloader):
            data, labels, private_labels = data.to(self.device), labels.to(self.device), private_labels.to(self.device)

            feature = fe_model(data)
            pred = classifier(feature)
            pred_private = mi_model(data, feature, private_labels)

            loss_utility = self.criterion(pred, labels)
            acc_utility = float(compute_metrics(pred, labels))
            loss_privacy = self.criterion(pred_private, private_labels)
            acc_privacy = float(compute_metrics(pred_private, private_labels))

            list_loss_utility.append(loss_utility.item())
            list_acc_utility.append(acc_utility)
            list_loss_privacy.append(loss_privacy.item())
            list_acc_privacy.append(acc_privacy)

        return sum(list_acc_utility)/len(list_acc_utility), sum(list_acc_privacy)/len(list_acc_privacy)


    def eval_test(self, fe_model, classifier, mi_model):
        fe_model.eval()
        classifier.eval()
        mi_model.eval()

        list_acc_utility, list_acc_privacy, list_loss_utility, list_loss_privacy = [], [], [], []

        for batch_idx, (data, labels, private_labels) in enumerate(self.testloader):
            data, labels, private_labels = data.to(self.device), labels.to(self.device), private_labels.to(self.device)

            feature = fe_model(data)
            pred = classifier(feature)
            pred_private = mi_model(data, feature, private_labels)

            # Necessary for JSD loss. x' is an random input data sampled independently of the same distribution of x
            aux1 = data[1:].clone()
            aux2 = data[0].unsqueeze(0).clone()
            x_prime = torch.cat((aux1, aux2), dim=0)

            loss_utility = self.criterion(pred, labels)
            acc_utility = float(compute_metrics(pred, labels))
            loss_privacy = -jsd_loss(mi_model, data, feature, private_labels, x_prime)
            #loss_privacy = self.criterion(pred_private, private_labels)
            acc_privacy = float(compute_metrics(pred_private, private_labels))

            list_loss_utility.append(loss_utility.item())
            list_acc_utility.append(acc_utility)
            list_loss_privacy.append(loss_privacy.item())
            list_acc_privacy.append(acc_privacy)

        return sum(list_acc_utility)/len(list_acc_utility), sum(list_acc_privacy)/len(list_acc_privacy), \
               sum(list_loss_utility)/len(list_loss_utility), sum(list_loss_privacy)/len(list_loss_privacy)

    def t_sne_plot(self, fe_model):
        fe_model.eval()
        targets_list, outputs_list = [], []

        for batch_idx, (data, labels, private_labels) in enumerate(self.plotloader):
            data, labels, private_labels = data.to(self.device), labels.to(self.device), private_labels.to(self.device)

            feature = fe_model(data)
            outputs_np = data.cpu().numpy()
            #outputs_np = feature.detach().cpu().numpy()
            targets_np = private_labels.detach().cpu().numpy()
            targets_list.append(targets_np[:, np.newaxis])
            outputs_list.append(outputs_np)

        targets = np.concatenate(targets_list, axis=0).astype(np.str)
        outputs = np.concatenate(outputs_list, axis=0).astype(np.float64)

        print('generating t-SNE plot...')
        #tsne = TSNE(random_state=0, n_iter=250, perplexity=200, learning_rate=100, early_exaggeration=10) #loans dataset
        tsne = TSNE(random_state=0, n_iter=250, perplexity=5, learning_rate=1200, early_exaggeration=10) #credit and adult income datasets

        scaled_data = StandardScaler().fit_transform(outputs)
        tsne_output = tsne.fit_transform(scaled_data)

        df = pd.DataFrame(tsne_output, columns=['x', 'y'])
        df['targets'] = targets

        plt.rcParams['figure.figsize'] = 10, 10
        sns.scatterplot(
            x=df['x'], y=df['y'],
            hue=df['targets'],
            palette=sns.color_palette('bright', len(df['targets'].unique())),
            marker='o',
            legend=False,
            linewidth=0
        )
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')

        plt.savefig(os.path.join('PPFL-v2/plots/loans/', 'race_rawdata.pdf'), bbox_inches='tight', format='pdf')

        # fig = px.scatter(df, x='x', y='y', color='targets')
        #fig.write_html(os.path.join('PPFL-v2/plots/credit', 'tsne_try.html'))

        print('done!')

