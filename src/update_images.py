#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy

import torch
import torchvision.transforms
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from utils import private_label_obtention, compute_metrics, jsd_MI
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
information=[]
from sklearn.preprocessing import StandardScaler

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)
        #return image.clone().detach(),label.clone().detach


class Pretrain(object):
    def __init__(self, args, dataset, logger):
        self.args = args
        self.logger = logger
        self.trainloader = DataLoader(dataset, batch_size=128, shuffle=False)
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
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                #obtaining private_labels
                private_labels = private_label_obtention(self.args, labels)
                private_labels = private_labels.type(torch.LongTensor)
                private_labels = private_labels.to(self.device)

                # The feature extractor is trained to accurately predict the real label
                feature = fe_model(images)
                pred = classifier(feature)
                loss = self.criterion(pred, private_labels)

                optimizer_fe.zero_grad()
                optimizer_classifier.zero_grad()
                loss.backward()
                optimizer_fe.step()
                optimizer_classifier.step()

        return fe_model, classifier

    def eval_pretrain_fe(self, fe_model, classifier):
        fe_model.eval()
        classifier.eval()

        with torch.no_grad():

            for batch_idx, (images, labels) in enumerate(self.trainloader):
                list_acc, list_loss = [], []
                images, labels = images.to(self.device), labels.to(self.device)

                private_labels = private_label_obtention(self.args, labels)
                private_labels = private_labels.type(torch.LongTensor)
                private_labels = private_labels.to(self.device)

                feature = fe_model(images)
                pred = classifier(feature)
                loss = self.criterion(pred, private_labels)
                acc = float(compute_metrics(pred, private_labels))
                list_loss.append(loss.item())
                list_acc.append(acc)

        print('Average loss on pretraining: {} | Average accuracy on pretraining: {}%'.format(round(sum(list_loss)/len(list_loss),3),round(sum(list_acc)*100/len(list_acc),3)))
    def Pretrain_MI(self,fe_model, classifier,mi_model):
        fe_model.train()
        classifier.train()
        mi_model.train()
        if self.args.optimizer == 'sgd':
            optimizer_mi = torch.optim.SGD(mi_model.parameters(), self.args.lr_mi_estimator,0.5)  # Should be a model.utility (create an utility model)

        elif self.args.optimizer == 'adam':
            optimizer_mi = torch.optim.Adam(mi_model.parameters(), self.args.lr_mi_estimator, weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                #Obtention of the private label
                private_labels = private_label_obtention(self.args, labels)
                private_labels = private_labels.type(torch.LongTensor)
                private_labels = private_labels.to(self.device)

                # Necessary for JSD loss. x' is an random input data sampled independently of the same distribution of x
                aux1 = images[1:].clone()
                aux2 = images[0].unsqueeze(0).clone()
                x_prime = torch.cat((aux1, aux2), dim=0)

                feature = fe_model(images).detach()
                pred = classifier(feature).detach()
                private_labels = private_labels.to(self.device)
                loss_mi=-jsd_MI(mi_model, images, feature, pred, x_prime)

                optimizer_mi.zero_grad()
                loss_mi.backward()
                optimizer_mi.step()

        return mi_model

    def test_Pretrain_mi(self,fe_model,classifier,mi_model):
        fe_model.eval()
        classifier.eval()
        mi_model.eval()
        avg_loss=0
        counter=0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                list_acc, list_loss = [], []
                images, labels = images.to(self.device), labels.to(self.device)

                aux1 = images[1:].clone()
                aux2 = images[0].unsqueeze(0).clone()
                x_prime = torch.cat((aux1, aux2), dim=0)

                private_labels = private_label_obtention(self.args, labels)
                private_labels = private_labels.type(torch.LongTensor)
                private_labels = private_labels.to(self.device)

                feature = fe_model(images).detach()
                pred = classifier(feature).detach()
                avg_loss += jsd_MI(mi_model,images,feature,pred,x_prime)
                counter= counter+1

            avg_loss/=counter

        print('Test Avg. Loss: %f' % avg_loss)

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader, self.plotloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        plotloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=128, shuffle=False)

        n=10 #Default for cifar
        if self.args.dataset == 'lfw':
            n=2

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/n), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/n), shuffle=False)
        return trainloader, validloader, testloader, plotloader

    def train_fe(self, fe_model, classifier, mi_model, global_round,current_lr):
        # Set mode to train models
        cnt=0
        loss=0
        info_loss=[]
        fe_model.train()
        classifier.train()
        mi_model.train()

        if self.args.optimizer == 'sgd':
            optimizer_fe = torch.optim.SGD(fe_model.parameters(), self.args.lr_fe, 0.5)
            optimizer_classifier = torch.optim.SGD(classifier.parameters(), self.args.lr_classifier, 0.5) #Should be model.privacy (create a privacy model)
            optimizer_mi = torch.optim.SGD(mi_model.parameters(), self.args.lr_mi_estimator, 0.5) #Should be a model.utility (create an utility model)

        elif self.args.optimizer == 'adam':
            optimizer_fe = torch.optim.Adam(fe_model.parameters(), lr=current_lr, weight_decay=1e-4)
            optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=current_lr, weight_decay=1e-4)
            optimizer_mi = torch.optim.Adam(mi_model.parameters(), lr=current_lr, weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                #Obtention of the private label
                private_labels = private_label_obtention(self.args, labels)
                private_labels = private_labels.type(torch.LongTensor)
                private_labels = private_labels.to(self.device)


                # Necessary for JSD loss. x' is an random input data sampled independently of the same distribution of x
                aux1 = images[1:].clone()
                aux2 = images[0].unsqueeze(0).clone()
                x_prime = torch.cat((aux1, aux2), dim=0)

                feature = fe_model(images)
                pred = classifier(feature)
                private_labels = private_labels.to(self.device)

                loss_classifier = self.criterion(pred, private_labels)
                loss_classifier2 = copy.copy(loss_classifier)
                JSD_loss = -jsd_MI(mi_model, images, feature, private_labels, x_prime)
                JSD_loss2 = copy.copy(JSD_loss)
                total_loss = -self.args.tradeoff_lambda*loss_classifier + (1-self.args.tradeoff_lambda)*JSD_loss

                # optimizing FE
                optimizer_fe.zero_grad()
                total_loss.backward(retain_graph=True)
                optimizer_fe.step()
                feature.detach()

                #optimizing classifier
                optimizer_classifier.zero_grad()
                loss_classifier2.backward(retain_graph=True)
                optimizer_classifier.step()
                pred.detach()

                #optimizing mutual_information
                optimizer_mi.zero_grad()
                JSD_loss2.backward()
                optimizer_mi.step()
                cnt+=1
                loss+=JSD_loss.item()
                #info_loss.append(JSD_loss.item())

                #if self.args.verbose and (batch_idx % 5 == 0):
                  #  print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tTotal loss: {:.6f}'.format(
                   #     global_round, iter, batch_idx * len(images),
                    #    len(self.trainloader.dataset),
                     #                       100. * batch_idx / len(self.trainloader), total_loss.item()))
                    #print('loss_classifier',JSD_loss.item())
                    #print(cnt)
                #self.logger.add_scalar('loss', total_loss.item())
        #print(info_loss[:])
        #print(len(info_loss))
        #print(loss/cnt)
        js_mi=loss/cnt

        return fe_model.state_dict(),js_mi

    def eval_train(self, fe_model, classifier, mi_model):
        fe_model.eval()
        classifier.eval()
        mi_model.eval()

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                list_acc_utility, list_acc_privacy, list_loss_utility, list_loss_privacy = [], [], [], []
                images, labels = images.to(self.device), labels.to(self.device)

                #Obtention of the private label
                private_labels = private_label_obtention(self.args, labels)
                private_labels = private_labels.type(torch.LongTensor)
                private_labels = private_labels.to(self.device)

                feature = fe_model(images)
                pred = classifier(feature)
                #pred_private = mi_model(images,feature,private_labels)

                loss_privacy = self.criterion(pred, private_labels)
                acc_privacy = float(compute_metrics(pred, private_labels))
                #loss_privacy = self.criterion(pred_private, private_labels)
                #acc_privacy = float(compute_metrics(pred_private, private_labels))

                #list_loss_utility.append(loss_utility.item())
                #list_acc_utility.append(acc_utility)
                list_loss_privacy.append(loss_privacy.item())
                list_acc_privacy.append(acc_privacy)

        return sum(list_acc_privacy)/len(list_acc_privacy)#,sum(list_acc_utility)/len(list_acc_utility),


    def eval_test(self, fe_model, classifier, mi_model):
        fe_model.eval()
        classifier.eval()
        #mi_model.eval()

        with torch.no_grad():

            for batch_idx, (images, labels) in enumerate(self.testloader):
                list_acc_utility, list_acc_privacy, list_loss_utility, list_loss_privacy = [], [], [], []
                images, labels = images.to(self.device), labels.to(self.device)

                #Obtention of the private label
                private_labels = private_label_obtention(self.args, labels)
                private_labels = private_labels.type(torch.LongTensor)
                private_labels = private_labels.to(self.device)

                feature = fe_model(images)
                pred = classifier(feature)
                #pred_private = mi_model(images,feature,private_labels)

                print('Evaluation on test')
                loss_privacy = self.criterion(pred, private_labels)
                acc_privacy = float(compute_metrics(pred, private_labels))
                #loss_privacy = self.criterion(pred_private, private_labels)
                #acc_privacy = float(compute_metrics(pred_private, private_labels))

                #list_loss_utility.append(loss_utility.item())
                #list_acc_utility.append(acc_utility)
                list_loss_privacy.append(loss_privacy.item())
                list_acc_privacy.append(acc_privacy)

        return sum(list_acc_privacy)/len(list_acc_privacy),sum(list_loss_privacy) / len(list_loss_privacy)

        #return sum(list_acc_utility)/len(list_acc_utility), sum(list_acc_privacy)/len(list_acc_privacy), \
               #sum(list_loss_utility) / len(list_loss_utility), sum(list_loss_privacy) / len(list_loss_privacy)

    def t_sne_plot(self, fe_model):
        fe_model.eval()
        targets_list, outputs_list = [], []

        for batch_idx, (images, labels) in enumerate(self.plotloader):
            images, labels = images.to(self.device), labels.to(self.device)
            # Obtention of the private label
            private_labels = private_label_obtention(self.args, labels)
            private_labels = private_labels.type(torch.LongTensor)
            private_labels = private_labels.to(self.device)

            feature = fe_model(images)
            outputs_np = images.cpu().numpy()
            #outputs_np = feature.detach().cpu().numpy()
            outputs_np = outputs_np.reshape((len(outputs_np), 32*32*3))
            targets_np = private_labels.detach().cpu().numpy()
            targets_list.append(targets_np[:, np.newaxis])
            outputs_list.append(outputs_np)

        targets = np.concatenate(targets_list, axis=0).astype(np.str)
        outputs = np.concatenate(outputs_list, axis=0).astype(np.float64)


        print('generating t-SNE plot...')
        tsne = TSNE(random_state=0)

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

        plt.savefig(os.path.join('PPFL-v2/plots/cifar/', 'rawdata.pdf'), bbox_inches='tight', format='pdf')

        # fig = px.scatter(df, x='x', y='y', color='targets')
        #fig.write_html(os.path.join('PPFL-v2/plots/credit', 'tsne_try.html'))

        print('done!')


