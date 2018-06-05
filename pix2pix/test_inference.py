import sys
import os
import torch
import numpy as np
from torch.autograd import Variable
from loader import get_train_loader

import time
from pix2pix.models.pix2pix_model import Pix2PixModel
from pix2pix.options.train_options import TrainOptions
#device = torch.device("cuda:4")


import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model

if __name__ == '__main__':
    opt = TrainOptions().parse()
    
    torch.cuda.set_device(6)
    torch.cuda.current_device()
        
    opt.input_nc = 6
    opt.gpu_ids = [6]
    opt.batchSize = 1
    
    print(torch.cuda.current_device())
    dataset = get_train_loader("index.p", train=False, batch_size=opt.batchSize, resize_size=128, return_all=True)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    model.eval()
    
    print("Setting up the model")
    model.load_networks(200)

    for i in range(dataset_size):
        x, y, z = dataset.dataset[i]
        for j in range(1, len(x)):
            data = {}
            data['A'] = torch.cat([x[0][None], y[j][None]], dim=1)
            data['B'] = x[j][None]
            data['A_paths'] = '.'
            model.set_input(data)
            model.forward()
            pic = np.transpose(model.fake_B.data.cpu().numpy()[0], (1, 2, 0))
            np.save('results/pic_{}_{}.npy'.format(i, j), pic)
        



