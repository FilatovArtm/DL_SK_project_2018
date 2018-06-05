import torch
import numpy as np
from torch.autograd import Variable

def LossGAWNN(X, generator, discriminator):
    return 