      
import os
import glob
import cv2
import netCDF4 as nc
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data.dataset import Dataset
import sys
import fnmatch
import json
from petrel_client.client import Client
import io
import random
client = Client(conf_path="~/petreloss.conf")



class APDataset(Dataset):
    def __init__(self,
                 GEMS_path='s3://atmospheric_chemistry//media/PJLAB\wangzhengyi/air/V2_crop/GEMS_V2/daily',
                 TROPOMI_path='s3://atmospheric_chemistry//media/PJLAB\wangzhengyi/air/V2_crop/TROPOMI_V2/daily',
                 GEOS_path='s3://atmospheric_chemistry//media/PJLAB\wangzhengyi/air/V2_crop/GEOS_V2',
                 split='train'
                 ):
        super(APDataset, self).__init__()
        self.split = split
        self.GEMS_path = GEMS_path
        self.TROPOMI_path = TROPOMI_path
        self.GEOS_path = GEOS_path
        self.annotation_lines = json.load(open('/mnt/petrelfs/zhaoxiangyu1/code/weather_prompt_new/dataset/co2_list.json'))

    def __len__(self):
        if self.split == 'train':
            return len(self.annotation_lines[:20000])
        else:
            return len(self.annotation_lines[20000:21000])
 
    def get_item(self, index):
        # print(index)
        #选择对应样本的日期
        annotation_line = self.annotation_lines[index]
        # print(annotation_line)
        #找到对应文件
        GEMS_flist=find_nc_file(self.GEMS_path, 'GEMS_'+annotation_line)
        TROPOMI_flist=find_nc_file(self.TROPOMI_path, 'TROPOMI_'+annotation_line)
        GEOS_flist=find_npy_file(self.GEOS_path, 'GEOS_'+annotation_line)

        GEMS = io.BytesIO(client.get(GEMS_flist))
        #处理GEMS文件
        VCD_GEMS = GEMS_process(GEMS)
        #找到T并处理GEOS-CF文件
        GEOS = io.BytesIO(client.get(GEOS_flist))
        GEOS_CF = GEOS_process(GEOS)

        #找到TROPOMI文件
        TROPOMI = io.BytesIO(client.get(TROPOMI_flist))
        #处理TROPOMI文件
        VCD_TROPOMI= TROPOMI_process(TROPOMI)

        GEMS_GEOS = torch.cat([VCD_GEMS, GEOS_CF], dim=0)

        return GEMS_GEOS, VCD_TROPOMI

    def __getitem__(self, index):
        GEMS_GEOS_IN, VCD_TROPOMI_OUT = self.get_item(index)
        random_index = random.randint(0, self.__len__()-1)
        C_GEMS_GEOS_IN, C_VCD_TROPOMI_OUT = self.get_item(random_index)
        VCD_TROPOMI_OUT = np.repeat(VCD_TROPOMI_OUT, 2, axis=0)
        C_VCD_TROPOMI_OUT = np.repeat(C_VCD_TROPOMI_OUT, 2, axis=0)
        deg_type = 'NO2'
        if self.split == 'train':
            batch = [C_GEMS_GEOS_IN, C_VCD_TROPOMI_OUT, GEMS_GEOS_IN, VCD_TROPOMI_OUT]
            return batch, deg_type
        else:
            batch = {'input_query_img1': C_GEMS_GEOS_IN, 'target_img1': C_VCD_TROPOMI_OUT,
                    'input_query_img2': GEMS_GEOS_IN, 'target_img2': VCD_TROPOMI_OUT}
            return batch, deg_type


def find_nc_file(folder_path, annotation_line):
    nc_files = os.path.join(folder_path, annotation_line+'.nc')
    return nc_files
def find_npy_file(folder_path, annotation_line):
    nc_files = os.path.join(folder_path, annotation_line+'.npy')
    return nc_files

def GEMS_process(GEMS):
    GEMS = nc.Dataset('inmemory.nc', memory=GEMS.read())
    VCD_GEMS=GEMS.variables['VCD'][:]
    VCD_GEMS [VCD_GEMS > 100] = np.nan
    VCD_GEMS = torch.from_numpy(np.array(VCD_GEMS)).type(torch.FloatTensor)
    VCD_GEMS=torch.div(torch.sub(VCD_GEMS,2.3587),4.0309)
    VCD_GEMS= torch.where(torch.isnan(VCD_GEMS),torch.tensor(0.0),VCD_GEMS)
    VCD_GEMS= VCD_GEMS[np.newaxis,:,:] 
    # VCD_GEMS=F.pad(VCD_GEMS, (0, 0,4,4))
    return VCD_GEMS
def GEOS_process(GEOS):
    GEOS = np.load(GEOS)
    GEOS = GEOS[:]
    GEOS_CF = torch.from_numpy(np.array(GEOS)).type(torch.FloatTensor)
    GEOS_CF=torch.div(torch.sub(GEOS_CF,6.7271),10.8637)
    GEOS_CF= GEOS_CF[np.newaxis,:,:] 
    # GEOS_CF=F.pad(GEOS_CF, (0, 0,4,4))
    return GEOS_CF
def TROPOMI_process(TROPOMI):
    TROPOMI = nc.Dataset('inmemory.nc', memory=TROPOMI.read())
    VCD_TROPOMI=TROPOMI.variables['VCD'][:]
    #将GEMS和TROPOMI数据中大于100的异常值去掉
    VCD_TROPOMI [VCD_TROPOMI > 100] = np.nan
    VCD_TROPOMI = torch.from_numpy(np.array(VCD_TROPOMI)).type(torch.FloatTensor)
    #归一化
    VCD_TROPOMI=torch.div(torch.sub(VCD_TROPOMI,1.5490),2.5403)
    #把GEMS和TROPOMI里的nan值替换为0
    VCD_TROPOMI= torch.where(torch.isnan(VCD_TROPOMI),torch.tensor(0.0),VCD_TROPOMI)
    #把[1400,800]变成[1,1400,800]  
    VCD_TROPOMI= VCD_TROPOMI[np.newaxis,:,:]
    #将影像从[1，1400,800]填充至[1,1408,800]，使其在卷积时不会发生错误
    # VCD_TROPOMI= F.pad(VCD_TROPOMI, (0, 0,4,4)) 
    return VCD_TROPOMI

    