import sys
sys.path.append('/home/aafilatov/GAWNN-pytorch')
sys.path.append('/home/aafilatov/pytorch-CycleGAN-and-pix2pix')
import os
import torch
import numpy as np
from torch.autograd import Variable
from loader import get_train_loader

import time
from models.pix2pix_model import Pix2PixModel
from options.train_options import TrainOptions

#torch.cuda.set_device(4)
#device = torch.device("cuda:4")


import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model

if __name__ == '__main__':
    opt = TrainOptions().parse()
        
    opt.input_nc = 6
    opt.gpu_ids = [4]
    opt.batchSize = 4
    
    dataset = get_train_loader("index.p", batch_size=opt.batchSize, resize_size=128)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

    
    model = create_model(opt)
    model.setup(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, (data_x, data_y) in enumerate(dataset):
            data = {}
            data['A'] = torch.cat([data_x[:, 0], data_y[:, 1]], dim=1)
            data['B'] = data_x[:, 1]
            data['A_paths'] = '.'
        
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()
            #print("iteration completed")

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize
               

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

