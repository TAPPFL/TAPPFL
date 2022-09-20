from torch import nn
import torch.nn.functional as F
import math


class FeatureExtractor_images(nn.Module):
    def __init__(self):
        super(FeatureExtractor_images, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # initialize weight
        self._initialize_weights()



    def forward(self, x):
        out = self.convnet(x)
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

class FeatureExtractor_credit(nn.Module):
    def __init__(self):
        super(FeatureExtractor_credit, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        out = self.net(x)
        return out

class FeatureExtractor_loans(nn.Module):
    def __init__(self):
        super(FeatureExtractor_loans, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        out = self.net(x)
        return out

class FeatureExtractor_adult_income(nn.Module):
    def __init__(self):
        super(FeatureExtractor_adult_income, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        out = self.net(x)
        return out