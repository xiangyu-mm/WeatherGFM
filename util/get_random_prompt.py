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

catalog = pd.read_csv('/mnt/petrelfs/zhaoxiangyu1/code/weather_prompt_new/dataset/output.csv', low_memory=False)
img_types = set(['vis','ir069','ir107','vil'])
events = catalog.groupby('id').filter(lambda x: img_types.issubset(set(x['img_type']))).groupby('id')
event_ids = list(events.groups.keys())[3000:9000]

def get_line_time(value):
    row = catalog.loc[catalog['id'] == value].iloc[0]
    time = row['file_name'].split('/')[1]
    return time
min_score = 10000
num = 0
for i in range(15):
    num = num+300
    random_num = random.randint(0,300)+num
    item = event_ids[random_num]
    i_rand = random.randint(36,48)
    marked_event = item
    marked_frame = i_rand
    target_event_time = get_line_time(marked_event)
    path1 = f"s3://sevir_pair/vil/{target_event_time}/{marked_event}.npy"
    path2 = f"s3://sevir_pair/ir069/{target_event_time}/{marked_event}.npy"
    path3 = f"s3://sevir_pair/ir107/{target_event_time}/{marked_event}.npy"
    path4 = f"s3://sevir_pair/vis/{target_event_time}/{marked_event}.npy"
    target_data = io.BytesIO(client.get(path1))
    target_data = np.load(target_data)
    frame_id = marked_frame-36
    data_list = [frame_id, frame_id+6, frame_id+12, frame_id+18, frame_id+24, frame_id+36]
    target_data = target_data[:, :, data_list]
    target_data = (target_data-zscore_normalizations_sevir['vil']['shift'])/zscore_normalizations_sevir['vil']['scale']
    np.save(f'/mnt/petrelfs/zhaoxiangyu1/data/sevir_prompt_random/vil/{marked_event}.npy', target_data, allow_pickle=True)
    
    target_data2 = io.BytesIO(client.get(path2))
    target_data2 = np.load(target_data2)
    target_data2 = target_data2[:, :, marked_frame]
    target_data2 = (target_data2-zscore_normalizations_sevir['ir069']['shift'])/zscore_normalizations_sevir['ir069']['scale']
    np.save(f'/mnt/petrelfs/zhaoxiangyu1/data/sevir_prompt_random/ir069/{marked_event}.npy', target_data2, allow_pickle=True)

    target_data3 = io.BytesIO(client.get(path3))
    target_data3 = np.load(target_data3)
    target_data3 = target_data3[:, :, marked_frame]
    target_data3 = (target_data3-zscore_normalizations_sevir['ir107']['shift'])/zscore_normalizations_sevir['ir107']['scale']
    np.save(f'/mnt/petrelfs/zhaoxiangyu1/data/sevir_prompt_random/ir107/{marked_event}.npy', target_data3, allow_pickle=True)

    target_data4 = io.BytesIO(client.get(path4))
    target_data4 = np.load(target_data4)
    target_data4 = target_data4[:, :, marked_frame]
    target_data4 = (target_data4-zscore_normalizations_sevir['vis']['shift'])/zscore_normalizations_sevir['vis']['scale']
    np.save(f'/mnt/petrelfs/zhaoxiangyu1/data/sevir_prompt_random/vis/{marked_event}.npy', target_data4, allow_pickle=True)