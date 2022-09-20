" Utility preservation neural network"
" This NN takes as inputs the learnt representation (r) and the private attribute (u) and the input (x), and tries to infer the label"

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
import torch
import math

class mi_estimator_cifar(nn.Module):
    def __init__(self, args):
        super(mi_estimator_cifar, self).__init__()
        self.convnet_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),  # 3 color channels in the input image, 6 output channels
            nn.BatchNorm2d(16),  # normalize
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.convnet_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.fcnet = nn.Sequential(
            #nn.Linear(2 * 512 * 30 + num_classes, 4096),
            nn.Linear(14849, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, args.num_classes_priv)
        )

        # initialize weight
        self._initialize_weights()

    def forward(self, x, r, u):

        out1 = self.convnet_1(x)
        out1 = self.convnet_2(out1)
        out1 = out1.view(out1.size(0), -1)

        out2 = self.convnet_2(r)
        out2 = out2.view(out2.size(0), -1)

        u = torch.unsqueeze(u, 1)

        out = torch.cat((out1, out2, u), dim=1)
        out = self.fcnet(out)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class mi_estimator_lfw(nn.Module):
    def __init__(self, args):
        super(mi_estimator_lfw, self).__init__()
        self.convnet_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),  # 3 color channels in the input image, 6 output channels
            nn.BatchNorm2d(16),  # normalize
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.convnet_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.fcnet = nn.Sequential(
            #nn.Linear(2 * 512 * 30 + num_classes, 4096),
            nn.Linear(52937, 4096), #too big, how do I narrow it down
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, args.num_classes_priv)
        )

        # initialize weight
        self._initialize_weights()

    def forward(self, x, r, u):

        out1 = self.convnet_1(x)
        out1 = self.convnet_2(out1)
        out1 = out1.view(out1.size(0), -1)

        out2 = self.convnet_2(r)
        out2 = out2.view(out2.size(0), -1)

        u = torch.unsqueeze(u, 1)

        out = torch.cat((out1, out2, u), dim=1)

        out = self.fcnet(out)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class mi_estimator_credit(nn.Module):
    def __init__(self, args):
        super(mi_estimator_credit, self).__init__()
        self.net1 = nn.Sequential(
            nn.Linear(9, 16),
            nn.BatchNorm1d(16),  # normalize
            nn.ReLU(inplace=True),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.net2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        self.fcnet = nn.Sequential(
            nn.Linear(1025, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, args.num_classes_priv)
        )

    def forward(self, x, r, u):
        out1 = self.net1(x)
        out1 = self.net2(out1)
        out1 = out1.view(out1.size(0), -1)

        out2 = self.net2(r)
        out2 = out2.view(out2.size(0), -1)

        u = torch.unsqueeze(u, 1)

        out = torch.cat((out1, out2, u), dim=1)
        out = self.fcnet(out)

        return out

class mi_estimator_loans(nn.Module):
    def __init__(self, args):
        super(mi_estimator_loans, self).__init__()
        self.net1 = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),  # normalize
            nn.ReLU(inplace=True),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.net2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        self.fcnet = nn.Sequential(
            nn.Linear(1025, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, args.num_classes_priv)
        )

    def forward(self, x, r, u):
        out1 = self.net1(x)
        out1 = self.net2(out1)
        out1 = out1.view(out1.size(0), -1)

        out2 = self.net2(r)
        out2 = out2.view(out2.size(0), -1)

        u = torch.unsqueeze(u, 1)

        out = torch.cat((out1, out2, u), dim=1)
        out = self.fcnet(out)

        return out

class mi_estimator_adult_income(nn.Module):
    def __init__(self, args):
        super(mi_estimator_adult_income, self).__init__()
        self.net1 = nn.Sequential(
            nn.Linear(12, 16),
            nn.BatchNorm1d(16),  # normalize
            nn.ReLU(inplace=True),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.net2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        self.fcnet = nn.Sequential(
            nn.Linear(1025, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, args.num_classes_priv)
        )

    def forward(self, x, r, u):
        out1 = self.net1(x)
        out1 = self.net2(out1)
        out1 = out1.view(out1.size(0), -1)

        out2 = self.net2(r)
        out2 = out2.view(out2.size(0), -1)

        u = torch.unsqueeze(u, 1)

        out = torch.cat((out1, out2, u), dim=1)
        out = self.fcnet(out)

        return out