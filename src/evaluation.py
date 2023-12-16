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

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

class evaluation_model(object):
    def __init__(self, args, dataset, logger):
        self.args = args
        self.logger = logger
        self.trainloader,self.testloader = self.train_test(dataset,trainset,testset)
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)
    def train_test(self,dataset,trainset,testset):
        ##return train and test from dataset
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                                  shuffle=True)
        testloader= torch.utils.data.DataLoader(testset, batch_size=128,
                                                  shuffle=True)
        return trainloader,testloader

    def train_classifier_eval(self,fe_model,classifier_eval):
        fe_model.eval()
        classifier_eval.train()
        optimizer_classifier_eval= torch.optim.Adam(classifier_eval.parameters(), 0.0001, weight_decay=1e-4)
        for iter in range(20):
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_classifier_eval.zero_grad()
                features=fe_model(images).detach()
                output=classifier_eval(features)
                loss=self.criterion(output,labels)
                loss.backward()
                optimizer_classifier_eval.step()
        return classifier_eval
    def test_classifier_eval(self,fe_model,classifier_eval):
        fe_model.eval()
        classifier_eval.eval()
        avg_loss=0
        avg_acc=0
        counter=0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.testloader):
                list_acc, list_loss = [], []
                images, labels = images.to(self.device), labels.to(self.device)

                feature = fe_model(images)
                output = classifier_eval(feature)
                avg_loss += self.criterion(output, labels).sum()
                pred = output.detach().max(1)[1]
                avg_acc += pred.eq(labels.view_as(pred)).sum()
                counter += 1
        avg_loss /= counter
        avg_loss = avg_loss.detach().cpu().item()
        avg_acc = float(avg_acc) / len(self.testloader)
        print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss, avg_acc))
        return avg_loss,avg_acc
#### This part for datasets in pandas dataframe format







