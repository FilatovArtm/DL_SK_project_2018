import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pickle
from loader import get_train_loader
import time

def plot_gallery(images, h, w, n_row=3, n_col=6):
    """Helper function to plot a gallery of portraits"""
    scale_const = 1.2
    plt.figure(figsize=(3 / scale_const * n_col, 3.4 / scale_const * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].numpy().transpose(1,2,0), cmap=plt.cm.gray, vmin=-1, vmax=1, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())

class Autoencoder4(nn.Module):
    def __init__(self):
        super(Autoencoder4, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, padding=0),
#             nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=2, padding=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        latent_code = self.encoder(x)
        reconstruction = self.decoder(latent_code)
        return reconstruction, latent_code
    
class Autoencoder8(nn.Module):
    def __init__(self):
        super(Autoencoder8, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, padding=0),
#             nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=2, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=2, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ConvTranspose2d(16, 16, 2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        latent_code = self.encoder(x)
        reconstruction = self.decoder(latent_code)
        return reconstruction, latent_code

class Autoencoder16(nn.Module):
    def __init__(self):
        super(Autoencoder16, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, padding=0),
#             nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=2, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=2, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ConvTranspose2d(16, 16, 2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        latent_code = self.encoder(x)
#         print(latent_code.shape)
        reconstruction = self.decoder(latent_code)
        return reconstruction, latent_code

class Net4Group2(nn.Module):
    """
        Network for predictions group 2 images based on group 1.
        input size: 4x4
        output size: 4x4
    """

    def __init__(self, layers=[1], bottelneck_size=32):
        super(Net4Group2, self).__init__()
        
        self.first_part = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.up_1 = nn.Sequential(
            nn.ConvTranspose2d(48, 64, 1, stride=2)
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(2, 2), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(2, 2), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=(2, 2), stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x, keypoints, embeddings):
        x = self.first_part(x)
        x = torch.cat((x, keypoints, embeddings), dim=1)
        x = self.up_1(x)
        x = self.conv_1(x)
        return x

class Net4Group3(nn.Module):
    """
        Network for predictions group 2 images based on group 1.
        input size: 4x4
        output size: 4x4
    """
    def __init__(self, layers=[1], bottelneck_size=32):
        super(Net4Group3, self).__init__()
        
        self.first_part = nn.Sequential(
            
            nn.Conv2d(3, 8, kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=(1, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        self.up_1 = nn.Sequential(
            nn.ConvTranspose2d(48, 64, 1, stride=2)
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(2, 2), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(2, 2), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=(2, 2), stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x, keypoints, embeddings):
        x = self.first_part(x)
#         print(x.shape)
        x = torch.cat((x, keypoints, embeddings), dim=1)
        x = self.up_1(x)
        x = self.conv_1(x)
        return x

    
class Net4Group4(nn.Module):
    """
        Network for predictions group 2 images based on group 1.
        input size: 4x4
        output size: 4x4
    """

    def __init__(self, layers=[1], bottelneck_size=32):
        super(Net4Group4, self).__init__()
        
        self.first_part = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            
            nn.ReLU(inplace=True),
        )

        self.up_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 1, stride=2)
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(2, 2), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(2, 2), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=(2, 2), stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x, keypoints, embeddings):
        x = self.first_part(x)
        x = torch.cat((x, keypoints, embeddings), dim=1)
        x = self.up_1(x)
        x = self.conv_1(x)
        return x

class Net8Group2(nn.Module):
    """
        Network for predictions group 2 images based on group 1.
        input size: 4x4
        output size: 4x4
    """

    def __init__(self, layers=[1], bottelneck_size=32):
        super(Net8Group2, self).__init__()
        
        self.first_part = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=2, padding=0),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True)
        )

        self.up_1 = nn.Sequential(
            nn.ConvTranspose2d(96, 64, 2, stride=2)
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(2, 2), padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(2, 2), padding=1),
            nn.ReLU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(2, 2), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(2, 2), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=(2, 2), stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x, keypoints, embeddings):
        x = self.first_part(x)
        x = torch.cat((x, keypoints, embeddings), dim=1)
        x = self.up_1(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

    
class Net8Group3(nn.Module):
    """
        Network for predictions group 2 images based on group 1.
        input size: 4x4
        output size: 4x4
    """

    def __init__(self, layers=[1], bottelneck_size=32):
        super(Net8Group3, self).__init__()
        
        self.first_part = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=2, padding=0),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(1, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up_1 = nn.Sequential(
            nn.ConvTranspose2d(96, 64, 2, stride=2)
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(2, 2), padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(2, 2), padding=1),
            nn.ReLU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(2, 2), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(2, 2), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=(2, 2), stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x, keypoints, embeddings):
        x = self.first_part(x)
#         print(x.shape)
        x = torch.cat((x, keypoints, embeddings), dim=1)
        x = self.up_1(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

class Net8Group4(nn.Module):
    """
        Network for predictions group 2 images based on group 1.
        input size: 4x4
        output size: 4x4
    """

    def __init__(self, layers=[1], bottelneck_size=32):
        super(Net8Group4, self).__init__()
        
        self.first_part = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=2, padding=0),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=2, padding=1),
#             nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up_1 = nn.Sequential(
            nn.ConvTranspose2d(96, 64, 2, stride=2)
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(2, 2), padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(2, 2), padding=1),
            nn.ReLU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(2, 2), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(2, 2), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=(2, 2), stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x, keypoints, embeddings):
        x = self.first_part(x)
        x = torch.cat((x, keypoints, embeddings), dim=1)
        x = self.up_1(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x