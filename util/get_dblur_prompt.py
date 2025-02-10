from petrel_client.client import Client
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

import warnings
from torch.utils.data import Dataset
import os
import sys
import time
from timeit import default_timer as timer
import einops
import pandas as pd
import io
import random
client = Client(conf_path="~/petreloss.conf")

zscore_normalizations_sevir = {
    'vil':{'scale':47.54,'shift':33.44},
    'ir069':{'scale':1174.68,'shift':-3683.58},
    'ir107':{'scale':2562.43,'shift':-1552.80},
    'lght':{'scale':0.60517,'shift':0.02990},
    'vis':{'scale':2259.96,'shift':-1347.91}
}

def _load_frames(file, type):
    file_num = file // 12
    step = file % 12
    ceph_root2 = 's3://gongjunchao/radar_deblur_4long/train/I10O12/sevir/EarthFormer/TimeStep'

    file_path = os.path.join(ceph_root2+str(step), f'{type}_{file_num}.npy')
    with io.BytesIO(client.get(file_path)) as f:
        frame_data = np.load(f)
    frame_data = frame_data*255
    frame_data = (frame_data-zscore_normalizations_sevir['vil']['shift'])/zscore_normalizations_sevir['vil']['scale']
    return frame_data

files = list(range(0, 20000))*12

random_index = random.randint(0, len(files)-1)
random_file = files[random_index]
context_blur_frame_data = _load_frames(random_file, type='pred')
context_gt_frame_data = _load_frames(random_file, type='tar')

num = 0
for i in range(20):
    random_index = random.randint(0, len(files)-1)
    random_file = files[random_index]
    context_blur_frame_data = _load_frames(random_file, type='pred')
    context_gt_frame_data = _load_frames(random_file, type='tar')
    np.save(f'/mnt/petrelfs/zhaoxiangyu1/data/deblur_prompt_random/blur/{random_index}.npy', context_blur_frame_data, allow_pickle=True)
    np.save(f'/mnt/petrelfs/zhaoxiangyu1/data/deblur_prompt_random/gt/{random_index}.npy', context_gt_frame_data, allow_pickle=True)