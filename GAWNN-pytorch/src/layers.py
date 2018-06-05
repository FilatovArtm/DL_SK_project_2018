import torch
import numpy as np
torch.cuda.set_device(5)
from torch.autograd import Variable

class ConvLayer(torch.nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride=1, pad=0):
        super(ConvLayer, self).__init__()
        self.conv_layer = torch.nn.Sequential(
            torch.nn.Conv2d(channels_in, channels_out, kernel_size, stride, pad),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(channels_out)
        ).cuda()
    def forward(self, X):
        return self.conv_layer(X)

class DeconvLayer(torch.nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride=1, pad=0):
        super(DeconvLayer, self).__init__()
        self.deconv_layer = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(channels_in, channels_out, kernel_size, stride, pad),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(channels_out),
        ).cuda()
    def forward(self, X):
        return self.deconv_layer(X)

class Flatten(torch.nn.Module):
    def forward(self, X):
        return X.view(X.shape[0], -1)

class GlobalAveragePooling(torch.nn.Module):
    def forward(self, X):
        return torch.mean(torch.mean(X, dim=3), dim=2)