
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

def process_sevir(raw_data_path, save_path):

    years = ['2018', '2019']
    frames = ['0101_0630','0701_1231']
    frames4ir = ['0101_0430', '0501_0831', '0901_1231']
    events_type = ['STORMEVENTS', 'RANDOMEVENTS']
    raw_url = raw_data_path
    tensor_url = save_path
    predict_frame = ['nowcast_training_000','nowcast_testing_000']
    vis_frame= ['SEVIR_VIS_RANDOMEVENTS_2018_0321_0331',
    'SEVIR_VIS_RANDOMEVENTS_2018_0401_0410',
    'SEVIR_VIS_RANDOMEVENTS_2018_0411_0420',
    'SEVIR_VIS_RANDOMEVENTS_2018_0421_0430',
    'SEVIR_VIS_RANDOMEVENTS_2018_0501_0510',
    'SEVIR_VIS_RANDOMEVENTS_2018_0511_0520',
    'SEVIR_VIS_RANDOMEVENTS_2018_0521_0531',
    'SEVIR_VIS_RANDOMEVENTS_2018_0601_0610',
    'SEVIR_VIS_RANDOMEVENTS_2018_0611_0620',
    'SEVIR_VIS_RANDOMEVENTS_2018_0621_0630',
    'SEVIR_VIS_RANDOMEVENTS_2018_0701_0710',
    'SEVIR_VIS_RANDOMEVENTS_2018_0711_0720',
    'SEVIR_VIS_RANDOMEVENTS_2018_0721_0731',
    'SEVIR_VIS_RANDOMEVENTS_2018_0801_0810',
    'SEVIR_VIS_RANDOMEVENTS_2018_0811_0820',
    'SEVIR_VIS_RANDOMEVENTS_2018_0821_0831',
    'SEVIR_VIS_RANDOMEVENTS_2018_0901_0910',
    'SEVIR_VIS_RANDOMEVENTS_2018_0911_0920',
    'SEVIR_VIS_RANDOMEVENTS_2018_0921_0930',
    'SEVIR_VIS_RANDOMEVENTS_2018_1001_1010',
    'SEVIR_VIS_RANDOMEVENTS_2018_1011_1020',
    'SEVIR_VIS_RANDOMEVENTS_2018_1021_1031',
    'SEVIR_VIS_RANDOMEVENTS_2018_1101_1110',
    'SEVIR_VIS_RANDOMEVENTS_2018_1111_1120',
    'SEVIR_VIS_RANDOMEVENTS_2018_1121_1130',
    'SEVIR_VIS_RANDOMEVENTS_2018_1201_1210',
    'SEVIR_VIS_RANDOMEVENTS_2018_1211_1220',
    'SEVIR_VIS_RANDOMEVENTS_2018_1221_1230',
    'SEVIR_VIS_STORMEVENTS_2018_0101_0131',
    'SEVIR_VIS_STORMEVENTS_2018_0201_0228',
    'SEVIR_VIS_STORMEVENTS_2018_0301_0331',
    'SEVIR_VIS_STORMEVENTS_2018_0401_0430',
    'SEVIR_VIS_STORMEVENTS_2018_0501_0531',
    'SEVIR_VIS_STORMEVENTS_2018_0601_0630',
    'SEVIR_VIS_STORMEVENTS_2018_0701_0731',
    'SEVIR_VIS_STORMEVENTS_2018_0801_0831',
    'SEVIR_VIS_STORMEVENTS_2018_0901_0930',
    'SEVIR_VIS_STORMEVENTS_2018_1001_1031',
    'SEVIR_VIS_STORMEVENTS_2018_1101_1130',
    'SEVIR_VIS_STORMEVENTS_2018_1201_1231']
    vis_frame2 = ['SEVIR_VIS_RANDOMEVENTS_2019_0101_0110',
    'SEVIR_VIS_RANDOMEVENTS_2019_0111_0120',
    'SEVIR_VIS_RANDOMEVENTS_2019_0121_0131',
    'SEVIR_VIS_RANDOMEVENTS_2019_0201_0210',
    'SEVIR_VIS_RANDOMEVENTS_2019_0211_0220',
    'SEVIR_VIS_RANDOMEVENTS_2019_0221_0228',
    'SEVIR_VIS_RANDOMEVENTS_2019_0301_0310',
    'SEVIR_VIS_RANDOMEVENTS_2019_0311_0320',
    'SEVIR_VIS_RANDOMEVENTS_2019_0321_0331',
    'SEVIR_VIS_RANDOMEVENTS_2019_0401_0410',
    'SEVIR_VIS_RANDOMEVENTS_2019_0411_0420',
    'SEVIR_VIS_RANDOMEVENTS_2019_0421_0430',
    'SEVIR_VIS_RANDOMEVENTS_2019_0501_0510',
    'SEVIR_VIS_RANDOMEVENTS_2019_0511_0520',
    'SEVIR_VIS_RANDOMEVENTS_2019_0521_0531',
    'SEVIR_VIS_RANDOMEVENTS_2019_0611_0620',
    'SEVIR_VIS_RANDOMEVENTS_2019_0621_0630',
    'SEVIR_VIS_RANDOMEVENTS_2019_0701_0710',
    'SEVIR_VIS_RANDOMEVENTS_2019_0711_0720',
    'SEVIR_VIS_RANDOMEVENTS_2019_0721_0731',
    'SEVIR_VIS_RANDOMEVENTS_2019_0801_0810',
    'SEVIR_VIS_RANDOMEVENTS_2019_0811_0820',
    'SEVIR_VIS_RANDOMEVENTS_2019_0821_0831',
    'SEVIR_VIS_RANDOMEVENTS_2019_0901_0910',
    'SEVIR_VIS_RANDOMEVENTS_2019_0911_0920',
    'SEVIR_VIS_RANDOMEVENTS_2019_0921_0930',
    'SEVIR_VIS_RANDOMEVENTS_2019_1001_1010',
    'SEVIR_VIS_RANDOMEVENTS_2019_1011_1020',
    'SEVIR_VIS_RANDOMEVENTS_2019_1021_1031',
    'SEVIR_VIS_RANDOMEVENTS_2019_1101_1110',
    'SEVIR_VIS_RANDOMEVENTS_2019_1111_1120',
    'SEVIR_VIS_RANDOMEVENTS_2019_1121_1130',
    'SEVIR_VIS_STORMEVENTS_2019_0101_0131',
    'SEVIR_VIS_STORMEVENTS_2019_0201_0228',
    'SEVIR_VIS_STORMEVENTS_2019_0301_0331',
    'SEVIR_VIS_STORMEVENTS_2019_0401_0430',
    'SEVIR_VIS_STORMEVENTS_2019_0501_0531',
    'SEVIR_VIS_STORMEVENTS_2019_0601_0630',
    'SEVIR_VIS_STORMEVENTS_2019_0701_0731',
    'SEVIR_VIS_STORMEVENTS_2019_0801_0831',
    'SEVIR_VIS_STORMEVENTS_2019_0901_0930',
    ]
    for task in ['ir069','ir107','vis','vil']: 
        if task == 'predict':
            frames = predict_frame
            for frame in frames:
                path = f'{raw_url}/data/processed/{frame}.h5.tar.gz'

        for year in years:
            if task == 'vis':
                if year=='2018':
                    vis_frames = vis_frame
                else:
                    vis_frames = vis_frame2
                for frame in vis_frames:
                    path = f'{raw_url}/data/{task}/{year}/{frame}.h5'
                    ff = h5py.File(path, 'r')
                    a,b = ff
                    list_id = ff[a]
                    list_value = ff[b]
                    for x,y in tqdm(zip(list_id, list_value)):
                        path = f"{tensor_url}/{b}/{year}/{x.decode('utf-8')}.npy"
                        np.save(path, y)
            elif task == 'vil':
                for frame in frames4ir:
                    for event in events_type:
                        path = f'{raw_url}/data/{task}/{year}/SEVIR_{task.upper()}_{event}_{year}_{frame}.h5'
                        print(path)
                        ff = h5py.File(path, 'r')
                        a,b = ff
                        list_id = ff[a]
                        list_value = ff[b]
                        for x,y in tqdm(zip(list_id, list_value)):
                            path = f"{tensor_url}/{b}/{year}/{x.decode('utf-8')}.npy"
                            np.save(path, y)
            else:
                for frame in frames:
                    for event in events_type:
                        path = f'{raw_url}/data/{task}/{year}/SEVIR_{task.upper()}_{event}_{year}_{frame}.h5'
                        print(path)
                        ff = h5py.File(path, 'r')
                        a,b = ff
                        list_id = ff[a]
                        list_value = ff[b]
                        for x,y in tqdm(zip(list_id, list_value)):
                            path = f"{tensor_url}/{b}/{year}/{x.decode('utf-8')}.npy"
                            np.save(path, y)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--raw_data", type=str, required=True, help="path of raw sevir")
    parser.add_argument("--save_path", type=str, required=True, help="path of processed sevir")
    args = parser.parse_args()
    process_sevir(args.raw_data, args.save_path)
