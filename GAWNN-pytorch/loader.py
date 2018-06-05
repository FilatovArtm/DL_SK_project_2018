import pickle
import os
import numpy as np
import PIL.Image
from multiprocessing.pool import ThreadPool
import cv2
import math
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
import torch

class DeepFashionDataset(Dataset):
    '''
    indicies не порядковые просто номера, прошедшие фильтрацию, а словарь: ключ - номер человека, значения - 
    лист из а) пусть к картинке б) координаты stickmena
    '''
    def __init__(self, index_path, 
                 train=True, shuffle=False, transform=None,
                 return_keys = ["imgs", "joints", "norm_imgs", "norm_joints"], return_all=False):
                
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
            
        self.basepath = os.path.dirname(index_path)
        self.train = train
        self.shuffle_ = shuffle
        self.return_keys = return_keys
        self.jo = self.index["joint_order"]
        self.indices = defaultdict(list)
        for i in range(len(self.index["train"])):
            if self._filter(i):
                man_idx = self.index['imgs'][i].split('/')[1].split('_')[0]
                self.indices[int(man_idx)].append([self.index['imgs'][i], self.index["joints"][i]])
                
        to_delete = []
        for key, value in self.indices.items():
            if len(value) == 1:
                to_delete.append(key)
        for key in to_delete:
            self.indices.pop(key)
        #self.indices = np.array([i for i in range(len(self.index["train"])) if self._filter(i)])
        #self.shuffle()
        

        self.transform = transform
        self.return_all = return_all
     
        
    def shuffle(self):
        self.batch_start = 0
        if self.shuffle_:
            np.random.shuffle(self.indices)
        
    def valid_joints(self, *joints):
        j = np.stack(joints)
        return (j >= 0).all()

    def load_img(self, path):
        img = PIL.Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        x = np.asarray(img, dtype = "uint8")
        if len(x.shape) == 2:
            x = np.expand_dims(x, -1)

        return x
    
    def make_joint_img(self, img_shape, jo, joints):
        # three channels: left, right, center
        scale_factor = img_shape[1] / 128
        thickness = int(3 * scale_factor)
        imgs = list()
        for i in range(3):
            imgs.append(np.zeros(img_shape[:2], dtype = "uint8"))

        body = ["lhip", "lshoulder", "rshoulder", "rhip"]
        body_pts = np.array([[joints[jo.index(part),:] for part in body]])
        if np.min(body_pts) >= 0:
            body_pts = np.int_(body_pts)
            cv2.fillPoly(imgs[2], body_pts, 255)

        right_lines = [
                ("rankle", "rknee"),
                ("rknee", "rhip"),
                ("rhip", "rshoulder"),
                ("rshoulder", "relbow"),
                ("relbow", "rwrist")]
        for line in right_lines:
            l = [jo.index(line[0]), jo.index(line[1])]
            if np.min(joints[l]) >= 0:
                a = tuple(np.int_(joints[l[0]]))
                b = tuple(np.int_(joints[l[1]]))
                cv2.line(imgs[0], a, b, color = 255, thickness = thickness)

        left_lines = [
                ("lankle", "lknee"),
                ("lknee", "lhip"),
                ("lhip", "lshoulder"),
                ("lshoulder", "lelbow"),
                ("lelbow", "lwrist")]
        for line in left_lines:
            l = [jo.index(line[0]), jo.index(line[1])]
            if np.min(joints[l]) >= 0:
                a = tuple(np.int_(joints[l[0]]))
                b = tuple(np.int_(joints[l[1]]))
                cv2.line(imgs[1], a, b, color = 255, thickness = thickness)

        rs = joints[jo.index("rshoulder")]
        ls = joints[jo.index("lshoulder")]
        cn = joints[jo.index("cnose")]
        neck = 0.5*(rs+ls)
        a = tuple(np.int_(neck))
        b = tuple(np.int_(cn))
        if np.min(a) >= 0 and np.min(b) >= 0:
            cv2.line(imgs[0], a, b, color = 127, thickness = thickness)
            cv2.line(imgs[1], a, b, color = 127, thickness = thickness)

        cn = tuple(np.int_(cn))
        leye = tuple(np.int_(joints[jo.index("leye")]))
        reye = tuple(np.int_(joints[jo.index("reye")]))
        if np.min(reye) >= 0 and np.min(leye) >= 0 and np.min(cn) >= 0:
            cv2.line(imgs[0], cn, reye, color = 255, thickness = thickness)
            cv2.line(imgs[1], cn, leye, color = 255, thickness = thickness)

        img = np.stack(imgs, axis = -1)
        if img_shape[-1] == 1:
            img = np.mean(img, axis = -1)[:,:,None]
        return img
    
    def _filter(self, i):
        good = True
        good = good and (self.index["train"][i] == self.train)
        joints = self.index["joints"][i]
        required_joints = ["lshoulder","rshoulder","lhip","rhip"]
        joint_indices = [self.jo.index(b) for b in required_joints]
        joints = np.float32(joints[joint_indices])
        good = good and self.valid_joints(joints)
        return good

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        
        
        idx = list(self.indices.keys())[i]
        
        imgs = list()
        stickmans = list()
        pos_vectors = list()
        
        joint_coords = []
        for num_man, man in enumerate(self.indices[idx]):
            img = self.load_img(man[0])
            width, height = img.shape[0], img.shape[1]
            joints_to_img_size = np.array([[width, height]])
            joint_coord = man[1] * joints_to_img_size
            joint_coords.append(torch.tensor(joint_coord))
            
            stickman = self.make_joint_img((width, height, 3), self.jo, joint_coord)
            if self.transform:
                img = self.transform(PIL.Image.fromarray(img))
                stickman = self.transform(PIL.Image.fromarray(stickman))
            imgs.append(img)
            stickmans.append(stickman)
        '''   
        #rel_img_path = self.index["imgs"][idx]
        #path = os.path.join(self.basepath, rel_img_path)
        img = self.load_img(path)
        
        width, height = img.shape[0], img.shape[1]
        joints_to_img_size = np.array([[[width, height]]])
        
        joint_coord = (self.index["joints"] * joints_to_img_size)[idx] 
        stickman = self.make_joint_img((width, height, 3), self.jo, joint_coord)
        
        if self.transform:
            img = self.transform(PIL.Image.fromarray(img))
            stickman = self.transform(PIL.Image.fromarray(stickman))
        '''    
       
        if self.return_all:
            return imgs, stickmans, joint_coords
        inds = np.random.choice(len(imgs), size=2, replace=False)
        return torch.stack([imgs[inds[0]], imgs[inds[1]]], 0), torch.stack([stickmans[inds[0]], stickmans[inds[1]]], 0)

def get_train_loader(index_path, batch_size=3, random_seed=42, shuffle=True, resize_size=None, train=True, return_all=False):
    transforms = []
    if resize_size:
        transforms.append(Resize(resize_size))
    transforms.append(ToTensor())
    train_transform = Compose(transforms)
    train_dataset = DeepFashionDataset(index_path, train=train, transform=train_transform, shuffle=shuffle, return_all=return_all)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    return train_loader

