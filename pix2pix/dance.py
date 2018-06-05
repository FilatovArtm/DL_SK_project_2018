import sys
sys.path.append('/home/aafilatov/GAWNN-pytorch')
sys.path.append('/home/aafilatov/pytorch-CycleGAN-and-pix2pix')
import os
import torch
import numpy as np
from torch.autograd import Variable
from loader import get_train_loader

import PIL.Image
import time
from models.pix2pix_model import Pix2PixModel
from options.train_options import TrainOptions
#device = torch.device("cuda:4")


import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model

initial_dance = np.array([[-256., -256.],
       [ 142.,   78.],
       [ 180.,   83.],
       [ 177.,  149.],
       [ 182.,  204.],
       [ 103.,   74.],
       [  99.,  145.],
       [  84.,  209.],
       [ 159.,  204.],
       [-256., -256.],
       [-256., -256.],
       [ 114.,  205.],
       [-256., -256.],
       [-256., -256.],
       [-256., -256.],
       [-256., -256.],
       [ 164.,   32.],
       [ 124.,   33.]])
def generate_circle_trajectory(radius, center, start_y, end_y):
    y = np.arange(int(np.min([start_y, end_y])), int(np.max([start_y, end_y])) + 1)
    
    x = np.round(center[0] + np.sqrt(radius ** 2 - (y - center[1]) ** 2)).astype(np.int32)
    return x

dummy = np.copy(initial_dance)
radius = np.round(np.linalg.norm(dummy[7] - dummy[6]) + 1)
y = np.arange(-radius, radius + 1) 
x = generate_circle_trajectory(radius, [0, 0], -radius, radius) / 1.5
dancing_poses = []

for i in range(len(x)):
    dummy = np.copy(initial_dance)
    dummy[7] = dummy[6] - np.array([x[i], y[i]])
    dummy[4] = dummy[3] - np.array([-x[i], y[i]])
    #plt.imshow(dataset.make_joint_img((256, 256, 3), dataset.jo, dummy))
    #plt.show()
    dancing_poses.append(dummy)

for i in range(len(x)):
    dummy = np.copy(initial_dance)
    dummy[7] = dummy[6] + np.array([x[i], y[i]])
    dummy[4] = dummy[3] + np.array([-x[i], y[i]])
    #plt.imshow(dataset.make_joint_img((256, 256, 3), dataset.jo, dummy))
    #plt.show()
    dancing_poses.append(dummy)

if __name__ == '__main__':
    opt = TrainOptions().parse()
    
    torch.cuda.set_device(6)
    torch.cuda.current_device()
        
    opt.input_nc = 6
    opt.gpu_ids = [6]
    opt.batchSize = 1
    
    print(torch.cuda.current_device())
    dataset = get_train_loader("index.p", train=False, batch_size=opt.batchSize, resize_size=128)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    
    print("Setting up the model")
    model.load_networks(200)
    data_x = dataset.dataset[9][0][None]

    for i in range(len(dancing_poses)):
        data_y = dataset.dataset.make_joint_img((256, 256, 3), dataset.dataset.jo, dancing_poses[i])
        data_y = dataset.dataset.transform(PIL.Image.fromarray(data_y))
        
        
        data = {}
        data['A'] = torch.cat([data_x[:, 0], data_y[None]], dim=1)
        data['B'] = data_x[:, 1]
        data['A_paths'] = '.'
        model.set_input(data)
        model.forward()
        pic = np.transpose(model.fake_B.data.cpu().numpy()[0], (1, 2, 0))
        np.save('results/dance/pic_{}.png'.format(i), pic)
        
