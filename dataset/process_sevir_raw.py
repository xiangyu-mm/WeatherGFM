
# # Creating manual catalogs, once again!
# ## Procedure is as follows:
# 
# 1) Filter for pct_missing==0
# 2) Filter for having 4 modalities as ['ir069','ir107','vil', 'lght']
# 
# 3) Drop duplicates
# 4) Divide for train & test
# 5) Summarise num of samples for both part!

import pandas as pd
import numpy as np
import h5py
import datetime
import gc
from tqdm import tqdm
import pickle

from argparse import ArgumentParser
from petrel_client.client import Client
import io

client = Client(conf_path="~/petreloss.conf")

parser = ArgumentParser()
parser.add_argument("-dataMod", "--dataMod", default="test", help="test or train data to work on",
    choices=["train", "test"])
parser.add_argument("-dirBase", "--dirBase", default="s3://sevir_raw",
    help="Base directory where all other data is kept in")
parser.add_argument("-dirOut", "--dirOut", default= "/mnt/petrelfs/zhaoxiangyu1/sevir_process/", 
    help="Sub directory where the data will be written!")
args = vars(parser.parse_args())

dataMod = args["dataMod"]
dirBase = args["dirBase"]
dirOut = args["dirOut"]

#### 0) Load the catalog!
path = f"{dirBase}/CATALOG.csv"
data = io.BytesIO(client.get(path))
catalog = pd.read_csv(data, low_memory=False)

# 1) Filter for pct_missing==0
cat_noZero = catalog[catalog['pct_missing']==0.0]
print("Events with no missing value: ", cat_noZero.shape)

# 2) Filter for having 4 modalities as ['ir069','ir107','vil', 'lght']
img_types = set(['vis','ir069','ir107','vil'])

# Group by event id, and filter to only events that have all desired img_types
events = cat_noZero.groupby('id').filter(lambda x: img_types.issubset(set(x['img_type']))).groupby('id')
event_ids = list(events.groups.keys())
print('Found %d events matching' % len(event_ids),img_types)

cat_full = cat_noZero[cat_noZero['id'].isin(event_ids)]
print("Events w/ 4 modalities: ", cat_full.shape)

# 3) Drop duplicates
cat_noZero_noDub = cat_full.drop_duplicates(subset=['id', 'img_type'])
print("Events with no dublicates: ", cat_noZero_noDub.shape)

df = pd.DataFrame(cat_noZero_noDub)
df.to_csv('output_day.csv', index=False)  # index=False 表示不保存索引列

# 4) Divide for train & test
train_cat_noZero_noDub = cat_noZero_noDub[cat_noZero_noDub['time_utc']<'2019-06-01']
test_cat_noZero_noDub = cat_noZero_noDub[cat_noZero_noDub['time_utc']>='2019-06-01']
print("Train events: ", train_cat_noZero_noDub.shape)
print("Test events: ", test_cat_noZero_noDub.shape)

# 5) Summarise and sanity check!

train_cat_noZero_noDub_lght = train_cat_noZero_noDub[train_cat_noZero_noDub['img_type']=='vis']
test_cat_noZero_noDub_lght = test_cat_noZero_noDub[test_cat_noZero_noDub['img_type']=='vis']
print("Train events w/ vis: ", train_cat_noZero_noDub_lght.shape)
print("Test events w/ vis: ", test_cat_noZero_noDub_lght.shape)

train_cat_noZero_noDub_ir069 = train_cat_noZero_noDub[train_cat_noZero_noDub['img_type']=='ir069']
test_cat_noZero_noDub_ir069 = test_cat_noZero_noDub[test_cat_noZero_noDub['img_type']=='ir069']
print("Train events w/ ir069: ", train_cat_noZero_noDub_ir069.shape)
print("Test events w/ ir069: ", test_cat_noZero_noDub_ir069.shape)

train_cat_noZero_noDub_vil = train_cat_noZero_noDub[train_cat_noZero_noDub['img_type']=='vil']
test_cat_noZero_noDub_vil = test_cat_noZero_noDub[test_cat_noZero_noDub['img_type']=='vil']
print("Train events w/ vil: ", train_cat_noZero_noDub_vil.shape)
print("Test events w/ vil: ", test_cat_noZero_noDub_vil.shape)

# train_cat_noZero_noDub_lght = train_cat_noZero_noDub[train_cat_noZero_noDub['img_type']=='lght']
# test_cat_noZero_noDub_lght = test_cat_noZero_noDub[test_cat_noZero_noDub['img_type']=='lght']
# print("Train events w/ lght: ", train_cat_noZero_noDub_lght.shape)
# print("Test events w/ lght: ", test_cat_noZero_noDub_lght.shape)




if dataMod == 'train':
    myCatalog =  cat_noZero_noDub[cat_noZero_noDub['time_utc']<'2019-06-01']
elif dataMod == 'test':
    myCatalog = cat_noZero_noDub[cat_noZero_noDub['time_utc']>='2019-06-01']

# Divide the catalog for convenience!
catalog_vis = myCatalog[myCatalog['img_type'] == 'vis'].reset_index()
catalog_ir069 = myCatalog[myCatalog['img_type'] == 'ir069'].reset_index()
catalog_ir107 = myCatalog[myCatalog['img_type'] == 'ir107'].reset_index()
# catalog_lght = myCatalog[myCatalog['img_type'] == 'lght'].reset_index()
catalog_vil = myCatalog[myCatalog['img_type'] == 'vil'].reset_index()


# # print(catalog_ir069.shape, catalog_ir107.shape, catalog_lght.shape, catalog_vil.shape)


# for i in tqdm(range(len(catalog_ir069))):
#     # 1) Read the name!
#     tmp_id = catalog_ir069['id'][i]

#     ## 2) Read relevant events!
#     # Prepare the catalogs
#     tmpCat_vis = catalog_vis.loc[catalog_vis['id'] == tmp_id].reset_index()
#     tmpCat_ir069 = catalog_ir069.loc[catalog_ir069['id'] == tmp_id].reset_index()
#     tmpCat_ir107 = catalog_ir107.loc[catalog_ir107['id'] == tmp_id].reset_index()
#     # tmpCat_lght = catalog_lght.loc[catalog_lght['id'] == tmp_id].reset_index()
    
#     tmpCat_vil = catalog_vil.loc[catalog_vil['id'] == tmp_id].reset_index()
#     buffer = io.BytesIO()

#     tensor_url = 's3://sevir_pair'
#     # Read the events
#     path = f"{dirBase}/data/{tmpCat_vis['file_name'][0]}"
#     data = io.BytesIO(client.get(path))
#     with h5py.File(data, 'r') as hf:
#         tmp_vis = hf['vis'][tmpCat_vis['file_index'][0]]
#     np.save(buffer, tmp_vis)
#     # 获取保存后的字节数据
#     bytes_data = buffer.getvalue()
#     path = f"{tensor_url}/test/{tmp_id}_vis.npy"
#     client.put(path, bytes_data)

#     buffer = io.BytesIO()
#     path = f"{dirBase}/data/{tmpCat_ir069['file_name'][0]}"
#     data = io.BytesIO(client.get(path))
#     with h5py.File(data, 'r') as hf:
#         tmp_ir069 = hf['ir069'][tmpCat_ir069['file_index'][0]]
#     path = f"{tensor_url}/test/{tmp_id}_ir069.npy"
#     np.save(buffer, tmp_ir069)
#     # 获取保存后的字节数据
#     bytes_data = buffer.getvalue()
#     client.put(path, bytes_data)

#     buffer = io.BytesIO()
#     path = f"{dirBase}/data/{tmpCat_ir107['file_name'][0]}"
#     data = io.BytesIO(client.get(path))
#     with h5py.File(data, 'r') as hf:
#         tmp_ir107 = hf['ir107'][tmpCat_ir107['file_index'][0]]
#     path = f"{tensor_url}/test/{tmp_id}_ir107.npy"
#     np.save(buffer, tmp_ir107)
#     # 获取保存后的字节数据
#     bytes_data = buffer.getvalue()
#     client.put(path, bytes_data)

#     buffer = io.BytesIO()
#     path = f"{dirBase}/data/{tmpCat_vil['file_name'][0]}"
#     data = io.BytesIO(client.get(path))
#     with h5py.File(data, 'r') as hf:
#         tmp_vil = hf['vil'][tmpCat_vil['file_index'][0]]
#     path = f"{tensor_url}/test/{tmp_id}_vil.npy"
#     np.save(buffer, tmp_vil)
#     # 获取保存后的字节数据
#     bytes_data = buffer.getvalue()
#     client.put(path, bytes_data)


