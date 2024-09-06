import h5py
import torch.utils.data as data
import numpy as np
import cv2
import random
import torch
from PIL import Image, ImageOps
import pandas as pd
import io
import os

from petrel_client.client import Client

client = Client(conf_path="~/petreloss.conf")
path_root = 'cluster2:s3://zwl2/RainNet_sp4/'

class RainNetDataset(data.Dataset):
    def __init__(self,split='train', frame=1, is_crop=True, crop_size_gt=512, upscale_factor=3, is_augment=False ,rotation=False, flip=False):
        # Note crop_size_gt in here is high_resolution image shape[crop_size_gt, crop_size_gt]
        super(RainNetDataset,self).__init__()
        self.name_list = RainNet_from_folder()
        self.split = split
        self.len = len(self.name_list)
        self.frame = frame
        self.is_crop    = is_crop
        self.crop_size_gt   = crop_size_gt
        self.upscale_factor = upscale_factor
        self.is_augment = is_augment
        self.rotation = rotation
        self.flip     = flip
        
        self.high_res_mean = 1.6412
        self.high_res_std = 0.7085
        self.low_res_mean = 0.5506
        self.low_res_std = 0.2375
        
        self.max = 140.0
        self.min = 0.0

    def get_item(self,index):
        file_path = path_root+self.name_list[index]+'.hdf5'
        data = io.BytesIO(client.get(file_path))
        f = h5py.File(data, 'r')
        img_hr = f['highres']
        img_lr = f['lowres']

        if self.is_crop:
            # random crop
            img_hr, img_lr = random_crop(img_hr, img_lr, self.crop_size_gt, self.upscale_factor)
            
        if self.is_augment:
            # flip, rotation if wanna augment
            img_hr, img_lr = augment([img_hr, img_lr], self.flip, self.rotation)
        
        # There change the batch_channel to shape[0]
        img_hr, img_lr = toTensor([img_hr, img_lr])
        
        # Mean_Std Norm
        # img_hr = (img_hr - self.high_res_mean) / self.high_res_std
        # img_lr = (img_lr - self.low_res_mean) / self.low_res_std
        
        # Max_min Norm:
        img_hr = (img_hr - self.min) / (self.max - self.min)
        img_lr = (img_lr - self.min) / (self.max - self.min)
        

        return img_hr, img_lr
    def __getitem__(self, index):
        
        id = index // 4
        frame_id = index % 4
        # 加载原数据
        file = self.get_item(id)
        random_index = random.randint(0, self.len-1)
        random_file = self.get_item(random_index)
        deg_type = 'lr_trans'
        img_hr, img_lr = file
        context_img_hr, context_img_lr = random_file
        img_hr = img_hr[frame_id,:,:]
        img_lr = img_lr[frame_id,:,:]
        context_img_hr = context_img_hr[frame_id,:,:]
        context_img_lr = context_img_lr[frame_id,:,:]
        # resize 256
        img_hr = np.expand_dims(cv2.resize(np.copy(img_hr), (256, 256),
                                interpolation=cv2.INTER_LINEAR), axis=2)
        img_lr = np.expand_dims(cv2.resize(np.copy(img_lr), (256, 256),
                                interpolation=cv2.INTER_LINEAR), axis=2)
        context_img_hr = np.expand_dims(cv2.resize(np.copy(context_img_hr), (256, 256),
                                interpolation=cv2.INTER_LINEAR), axis=2)
        context_img_lr = np.expand_dims(cv2.resize(np.copy(context_img_lr), (256, 256),
                                interpolation=cv2.INTER_LINEAR), axis=2)
        # formate channel
        img_lr = np.repeat(torch.from_numpy(np.ascontiguousarray(np.transpose(img_lr, (2, 0, 1)))).float(), 2, axis=0)
        img_hr = np.repeat(torch.from_numpy(np.ascontiguousarray(np.transpose(img_hr, (2, 0, 1)))).float(), 2, axis=0)
        context_img_lr = np.repeat(torch.from_numpy(np.ascontiguousarray(np.transpose(context_img_lr, (2, 0, 1)))).float(), 2, axis=0)
        context_img_hr = np.repeat(torch.from_numpy(np.ascontiguousarray(np.transpose(context_img_hr, (2, 0, 1)))).float(), 2, axis=0)


        if self.split == 'train':
            batch = [context_img_lr, context_img_hr, img_lr, img_hr]
            return batch, deg_type
        else:
            batch = {'input_query_img1': context_img_lr, 'target_img1': context_img_hr,
                    'input_query_img2': img_lr, 'target_img2': img_hr}
            return batch, deg_type

    def __len__(self):
        return self.len
    
# def RainNet_from_folder(folder):
#     h5_path = folder
#     f = h5py.File(h5_path,'r')
#     highres = f['highres']
#     lowres = f['lowres']
#     return highres,lowres
def RainNet_from_folder():
    df = pd.read_csv('/mnt/petrelfs/zhaoxiangyu1/code/weather_prompt_new/dataset/Rainnet.csv')
    name_list = df['name'].tolist()

    return name_list

# This is from original code (Doing random_crop to given shape while upscale_factor=3)
def random_crop(GT_img, LR_img, GT_crop_size,upscale_factor):

    LR_h, LR_w = LR_img.shape[0],LR_img.shape[1]
    LR_crop_size = GT_crop_size//upscale_factor

    ix = np.random.randint(0,LR_h-LR_crop_size-1)
    iy = np.random.randint(0,LR_w-LR_crop_size-1)

    ix_lr = ix//upscale_factor
    iy_lr = iy//upscale_factor

    GT_img_crop = GT_img[ix:ix+GT_crop_size, iy:iy+GT_crop_size,:]
    LR_img_crop = LR_img[ix_lr:ix_lr+LR_crop_size, iy_lr:iy_lr+LR_crop_size,:]

    return GT_img_crop, LR_img_crop

def augment(imgs, hflip=True, rotation=True):
    #print(imgs.shape)
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot = rotation and random.random() < 0.5

    def _augument(img):
        if hflip:
            cv2.flip(img, 1, img)
        if vflip:
            cv2.flip(img, 0, img)
        if rot:
            img = img.transpose(1, 0, 2)

        return img
    imgs = [_augument(img) for img in imgs]
    
    return imgs

def toTensor(imgs):

    def _totensor(img):
        img = torch.from_numpy(img.transpose(2, 0, 1))

        img = img.float()
        return img
    if isinstance(imgs, list):
        return [_totensor(img) for img in imgs]
    else:
        return _totensor(imgs)