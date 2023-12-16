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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models.classifier_evaluation import Classifier_image_cifar_evaluation
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms

dataset = pd.read_csv('/home/cc/PPFL-v2/data/loans/mortgageloans.csv')
trainset, testset = train_test_split(dataset, test_size=0.2)


class CustomDS(Dataset):
    def __init__(self, data_split):
        self.Y = torch.tensor(
            data_split['Affordability'].values,
            dtype=torch.float32
        )
        self.X = torch.tensor(
            data_split.drop(columns=['Affordability', 'Race'], axis=1).values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

trainDS = CustomDS(trainset)
testDS = CustomDS(testset)


class evaluation_loans(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.trainloader=DataLoader(trainDS,batch_size=128,shuffle=False)
        self.testloader=DataLoader(testDS,batch_size=128,shuffle=False)
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_classifier_eval(self,fe_model,classifier_eval):
        fe_model.eval()
        classifier_eval.train()
        optimizer_classifier_eval= torch.optim.Adam(classifier_eval.parameters(), 0.001, weight_decay=1e-4)
        for iter in range(5):
            for batch_idx, (data, labels) in enumerate(self.trainloader):
                labels = labels.type(torch.LongTensor)
                data, labels = data.to(self.device), labels.to(self.device)

                optimizer_classifier_eval.zero_grad()
                features=fe_model(data).detach()
                output=classifier_eval(features)
                #print(output,labels)
                loss=self.criterion(output,labels)
                loss.backward()
                optimizer_classifier_eval.step()
        return classifier_eval

    def test_classifier_eval(self, fe_model, classifier_eval):
        fe_model.eval()
        classifier_eval.eval()
        avg_loss = 0
        avg_acc = 0
        counter = 0
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(self.testloader):
                list_acc, list_loss = [], []
                labels = labels.type(torch.LongTensor)
                data, labels = data.to(self.device), labels.to(self.device)

                feature = fe_model(data)
                output = classifier_eval(feature)
                avg_loss += self.criterion(output, labels).sum()
                pred = output.detach().max(1)[1]
                avg_acc += pred.eq(labels.view_as(pred)).sum()
                counter += 1
        avg_loss /= counter
        avg_loss = avg_loss.detach().cpu().item()
        avg_acc = float(avg_acc) / len(self.testloader.dataset)
        print(len(self.testloader))
        print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss, avg_acc))
        return avg_loss, avg_acc

