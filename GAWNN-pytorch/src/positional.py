import torch
import numpy as np
from torch.autograd import Variable
torch.cuda.set_device(5)


class PositionTrasnformer:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def __call__(self, keypoints):
        maps = np.zeros((
            keypoints.shape[0], 
            keypoints.shape[1], 
            self.hidden_size, 
            self.hidden_size
            ))
        keypoints = keypoints.data.cpu().numpy()

        for i in range(maps.shape[0]):
            maps[i, 
            np.arange(maps.shape[1]), 
            keypoints[i, :, 0],
            keypoints[i, :, 1]] = 1

        return Variable(torch.from_numpy(maps).type(torch.FloatTensor)).cuda()
