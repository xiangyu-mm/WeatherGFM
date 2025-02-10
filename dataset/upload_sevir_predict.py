from datetime import datetime
import numpy as np
import os
import pandas as pd
import torch
import random
import io
import json
from tqdm import tqdm
import h5py
import tarfile

from petrel_client.client import Client

from dataset.upload_era5 import era5_128x256
import numpy as np

client = Client(conf_path="~/petreloss.conf")

raw_url = 's3://sevir_raw'
frame = 'nowcast_testing_000'
path = f'{raw_url}/data/processed/{frame}.h5.tar.gz'
data = io.BytesIO(client.get(path))

tar = tarfile.open(fileobj=data, mode='r:gz')
h5_file = tar.getmembers()

h5_filename = h5_file[0]
h5_file_data = tar.extractfile(h5_filename).read()

temp_h5_stream = io.BytesIO(h5_file_data)
f = h5py.File(temp_h5_stream, 'r')

a,b,c,d,e = f

tensor_url = 'cluster2:s3://zwl2/sevir_predict_test/'

for i in tqdm(range(len(f[d]))):
    buffer = io.BytesIO()
    path = f"{tensor_url}/{d}/{i}.npy"
    np.save(buffer, f[d][i])
    # 获取保存后的字节数据
    bytes_data = buffer.getvalue()
    client.put(path, bytes_data)

for i in tqdm(range(len(f[e]))):
    buffer = io.BytesIO()
    path = f"{tensor_url}/{e}/{i}.npy"
    np.save(buffer, f[e][i])
    # 获取保存后的字节数据
    bytes_data = buffer.getvalue()
    client.put(path, bytes_data)