import torch
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from datautil.datasplit import getdataloader
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_data(data_name):
    """Return the algorithm class with the given name."""
    datalist = {'medmnist': 'medmnist', 'medmnistA': 'medmnist'}
    if datalist[data_name] not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(data_name))
    return globals()[datalist[data_name]]


def gettransforms():
    transform_train = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])
    return transform_train, transform_test


class mydataset(object):
    def __init__(self, args):
        self.x = None
        self.targets = None
        self.dataset = None
        self.transform = None
        self.target_transform = None
        self.loader = None
        self.args = args

    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def __getitem__(self, index):
        x = self.input_trans(self.loader(self.x[index]))
        ctarget = self.target_trans(self.targets[index])
        return x, ctarget

    def __len__(self):
        return len(self.targets)


class MedMnistDataset(Dataset):
    def __init__(self, filename='', transform=None):
        self.data = np.load(filename+'xdata.npy')
        self.targets = np.load(filename+'ydata.npy')
        self.targets = np.squeeze(self.targets)
        self.transform = transform

        self.data = torch.Tensor(self.data)
        self.data = torch.unsqueeze(self.data, dim=1)

    def __len__(self):
        self.filelength = len(self.targets)
        return self.filelength

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def getlabeldataloader(args, data):
    trl, val, tel = getdataloader(args, data)
    trd, vad, ted = [], [], []
    for i in range(len(trl)):
        trd.append(torch.utils.data.DataLoader(
            trl[i], batch_size=args.batch, shuffle=True))
        vad.append(torch.utils.data.DataLoader(
            val[i], batch_size=args.batch, shuffle=False))
        ted.append(torch.utils.data.DataLoader(
            tel[i], batch_size=args.batch, shuffle=False))
    return trd, vad, ted

def medmnist(args):
    data = MedMnistDataset(args.root_dir+args.dataset+'/')
    trd, vad, ted = getlabeldataloader(args, data)
    args.num_classes = 11
    return trd, vad, ted

def medmnist_w(args):
    data = MedMnistDataset(args.root_dir+args.dataset+'/')
    args.num_classes = 11
    return data

def get_whole_dataset(data_name):
    datalist = {'medmnist': 'medmnist_w', 'medmnistA': 'medmnist_w'}
    if datalist[data_name] not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(data_name))
    return globals()[datalist[data_name]]
