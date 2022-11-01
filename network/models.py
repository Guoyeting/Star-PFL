import torch
import torch.nn as nn
import torch.nn.functional as func
from collections import OrderedDict

class lenet5v(nn.Module):
    def __init__(self):
        super(lenet5v, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 11)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        return y

    def getallfea(self, x):
        fealist = []
        y = self.conv1(x)
        fealist.append(y.clone().detach())
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        fealist.append(y.clone().detach())
        return fealist

    def getfinalfea(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        return [y]

    def get_sel_fea(self, x, plan=0):
        if plan == 0:
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = x.view(x.shape[0], -1)
            fealist = x
        elif plan == 1:
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = x.view(x.shape[0], -1)
            x = self.relu3(self.fc1(x))
            x = self.relu4(self.fc2(x))
            fealist = x
        else:
            fealist = []
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = x.view(x.shape[0], -1)
            fealist.append(x)
            x = self.relu3(self.fc1(x))
            fealist.append(x)
            x = self.relu4(self.fc2(x))
            fealist.append(x)
            fealist = torch.cat(fealist, dim=1)
        return fealist
