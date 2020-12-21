import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
from util.util import *
from math import floor
import math
from pytorch_wavelets import DWTForward, DWTInverse
###############################################################################
# Functions
###############################################################################

def weights_init_normal(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_zeros(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1 and classname != 'ConvLstmCell':

        init.uniform_(m.weight.data, 0.0, 0.0001)
        init.constant_(m.bias.data, 0.0)

    elif classname.find('Linear') != -1:
        init.uniform_(m.weight.data, 0.0, 0.0001)
        init.constant_(m.bias.data, 0.0)

    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 0.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_wavenet(m):
    classname = m.__class__.__name__

    # pdb.set_trace()
    if classname.find('Conv') != -1 and classname != 'ConvLstmCell':
        init.xavier_normal_(m.weight.data, gain=1)
        # init.constant(m.weight.data, 0.0)
        init.constant_(m.bias.data, 0.0)
        # m.weight.data = m.weight.data.double()
        # m.bias.data = m.bias.data.double()
        # pdb.set_trace()
    elif classname.find('Linear') != -1:
        init.uniform_(m.weight.data, 0.0, 0.02)
        init.constant_(m.bias.data, 0.0)
        # m.weight.data = m.weight.data.double()
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 0.0, 0.02)
        init.constant_(m.bias.data, 0.0)
        # m.weight.data = m.weight.data.double()
        # m.bias.data = m.bias.data.double()


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    elif init_type == 'wavenet':
        net.apply(weights_init_wavenet)
    elif init_type == 'zeros':
        net.apply(weights_init_zeros)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        
 
 
class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 *growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,padding=1)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out
        
        
        
class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 *growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,padding=1)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out
        
        
        
        
 class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1)
        self.avg_pool = nn.AvgPool2d(2)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.avg_pool(out)
        return out
