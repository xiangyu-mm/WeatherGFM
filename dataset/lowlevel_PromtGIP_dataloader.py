import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

from evaluate.add_degradation_various import *
from dataset.image_operators import *

import warnings
from torch.utils.data import Dataset
import os
import sys
import time
from timeit import default_timer as timer
import einops

from dataset.data_utiles import *
import dataset.util as util
import pandas as pd

from petrel_client.client import Client

client = Client(conf_path="~/petreloss.conf")

zscore_normalizations_sevir = {
    'vil':{'scale':47.54,'shift':33.44},
    'ir069':{'scale':1174.68,'shift':-3683.58},
    'ir107':{'scale':2562.43,'shift':-1552.80},
    'lght':{'scale':0.60517,'shift':0.02990},
    'vis':{'scale':2259.96,'shift':-1347.91}
}

# todo: ir609->ir107; 插帧60min->30min; 预测，4frame->1frame

def add_degradation_two_images(img_HQ1, img_HQ2, deg_type):
    if deg_type == 'GaussianNoise':
        level = random.uniform(10, 50)
        img_LQ1 = add_Gaussian_noise(img_HQ1.copy(), level=level)
        level = random.uniform(10, 50)
        img_LQ2 = add_Gaussian_noise(img_HQ2.copy(), level=level)
    elif deg_type == 'GaussianBlur':
        sigma = random.uniform(2, 4)
        img_LQ1 = iso_GaussianBlur(img_HQ1.copy(), window=15, sigma=sigma)
        sigma = random.uniform(2, 4)
        img_LQ2 = iso_GaussianBlur(img_HQ2.copy(), window=15, sigma=sigma)
    elif deg_type == 'JPEG':
        level = random.randint(10, 40)
        img_LQ1 = add_JPEG_noise(img_HQ1.copy(), level=level)
        level = random.randint(10, 40)
        img_LQ2 = add_JPEG_noise(img_HQ2.copy(), level=level)
    elif deg_type == 'Resize':
        img_LQ1 = add_resize(img_HQ1.copy())
        img_LQ2 = add_resize(img_HQ2.copy())
    elif deg_type == 'Rain':
        value = random.uniform(40, 200)
        img_LQ1 = add_rain(img_HQ1.copy(), value=value)
        value = random.uniform(40, 200)
        img_LQ2 = add_rain(img_HQ2.copy(), value=value)
    elif deg_type == 'SPNoise':
        img_LQ1 = add_sp_noise(img_HQ1.copy())
        img_LQ2 = add_sp_noise(img_HQ2.copy())
    elif deg_type == 'LowLight':
        lum_scale = random.uniform(0.3, 0.4)
        img_LQ1 = low_light(img_HQ1.copy(), lum_scale=lum_scale)
        img_LQ2 = low_light(img_HQ2.copy(), lum_scale=lum_scale)
    elif deg_type == 'PoissonNoise':
        img_LQ1 = add_Poisson_noise(img_HQ1.copy(), level=2)
        img_LQ2 = add_Poisson_noise(img_HQ2.copy(), level=2)
    elif deg_type == 'Ringing':
        img_LQ1 = add_ringing(img_HQ1.copy())
        img_LQ2 = add_ringing(img_HQ2.copy())
    elif deg_type == 'r_l':
        img_LQ1 = r_l(img_HQ1.copy())
        img_LQ2 = r_l(img_HQ2.copy())
    elif deg_type == 'Inpainting':
        l_num = random.randint(5, 10)
        l_thick = random.randint(5, 10)
        img_LQ1 = inpainting(img_HQ1.copy(), l_num=l_num, l_thick=l_thick)
        img_LQ2 = inpainting(img_HQ2.copy(), l_num=l_num, l_thick=l_thick)
    elif deg_type == 'gray':
        img_LQ1 = cv2.cvtColor(img_HQ1.copy(), cv2.COLOR_BGR2GRAY)
        img_LQ1 = np.expand_dims(img_LQ1, axis=2)
        img_LQ1 = np.concatenate((img_LQ1, img_LQ1, img_LQ1), axis=2)
        img_LQ2 = cv2.cvtColor(img_HQ2.copy(), cv2.COLOR_BGR2GRAY)
        img_LQ2 = np.expand_dims(img_LQ2, axis=2)
        img_LQ2 = np.concatenate((img_LQ2, img_LQ2, img_LQ2), axis=2)
    elif deg_type == '':
        img_LQ1 = img_HQ1
        img_LQ2 = img_HQ2
    elif deg_type == 'Laplacian':
        img_LQ1 = img_HQ1.copy()
        img_HQ1 = Laplacian_edge_detector(img_HQ1.copy())
        img_LQ2 = img_HQ2.copy()
        img_HQ2 = Laplacian_edge_detector(img_HQ2.copy())
    elif deg_type == 'Canny':
        img_LQ1 = img_HQ1.copy()
        img_HQ1 = Canny_edge_detector(img_HQ1.copy())
        img_LQ2 = img_HQ2.copy()
        img_HQ2 = Canny_edge_detector(img_HQ2.copy())
    elif deg_type == 'L0_smooth':
        img_LQ1 = img_HQ1.copy()
        img_HQ1 = L0_smooth(img_HQ1.copy())
        img_LQ2 = img_HQ2.copy()
        img_HQ2 = L0_smooth(img_HQ2.copy())
    else:
        print('Error!')
        exit()

    img_LQ1 = np.clip(img_LQ1*255, 0, 255).round().astype(np.uint8)
    img_LQ1 = img_LQ1.astype(np.float32)/255.0
    
    img_LQ2 = np.clip(img_LQ2*255, 0, 255).round().astype(np.uint8)
    img_LQ2 = img_LQ2.astype(np.float32)/255.0
    
    img_HQ1 = np.clip(img_HQ1*255, 0, 255).round().astype(np.uint8)
    img_HQ1 = img_HQ1.astype(np.float32)/255.0
    
    img_HQ2 = np.clip(img_HQ2*255, 0, 255).round().astype(np.uint8)
    img_HQ2 = img_HQ2.astype(np.float32)/255.0

    return img_LQ1, img_LQ2, img_HQ1, img_HQ2

class DatasetLowlevel_Train(Dataset):
    def __init__(self, dataset_path, input_size, phase, ITS_path=None, LOL_path=None, Rain13K_path=None,
                 GoPro_path=None, FiveK_path=None, LLF_path=None, RealOld_path=None, preprocess_OPERA=None,
                 size_target_center=None, full_opera_context=None, preprocess_HRIT=None, splits_path=None,
                 path_to_sample_ids='', years=None, split='training',**kwargs):
        
        np.random.seed(5)
        self.split=split
        self.HQ_size = input_size
        self.phase = phase
        w4c_predict = True
        # base dataset
        self.paths_HQ, self.sizes_HQ = util.get_image_paths('img', dataset_path)
        # self.paths_base_len = len(self.paths_HQ)
        print('dataset for test: ', len(self.paths_HQ))

        if w4c_predict:
            # self.regions = ['roxi_0004', 'roxi_0005', 'roxi_0006', 'roxi_0007']
            self.regions = ['roxi_0004']
            self.data_split = split
            self.input_product = "REFL-BT"
            self.output_product = "RATE"
            self.len_seq_in=1
            self.len_seq_predict=1
            self.splits_df = load_timestamps(splits_path)
            self.swap_time_ch=True
            self.years=years
            self.shuffle=False
            self.padding=0
            self.static_data=False
            self.max_samples=None
            self.stats_path=None
            self.static_root=''
            self.generate_samples=False
            self.sat_bands = ['IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073']
            # self.sat_bands = ['IR_108', 'VIS006', 'VIS008', 'WV_073']
            self.crop_edge = None
            self.out_min=[0.01]
            self.out_max=[1]
            self.full_opera_context=1512 # (105+42+105)*6; 
            self.size_target_center=252 # 42*6; 
            self.data_root='s3://w4c/w4c23/stage-2'
            # self.data_root='/mnt/petrelfs/zhaoxiangyu1/data/w4c'
            self.preprocess_target = preprocess_OPERA
            self.size_target_center = size_target_center
            self.full_opera_context = full_opera_context
            self.crop = int((self.full_opera_context - self.size_target_center) / 2) #calculate centre of image to begin crop
            self.preprocess_input = preprocess_HRIT
            self.path_to_sample_ids = path_to_sample_ids

            self.idxs = load_sample_ids(
                self.data_split, 
                self.splits_df,
                self.len_seq_in, 
                self.len_seq_predict, 
                self.regions,
                self.generate_samples, 
                self.years, 
                self.path_to_sample_ids)
            
                    
            self.in_ds = load_dataset(self.data_root, self.data_split, self.regions, self.years, self.input_product) #LOAD DATASET

            if self.data_split not in ['test', 'heldout']:
                self.out_ds = load_dataset(self.data_root, self.data_split, self.regions, self.years, self.output_product)
            else:
                self.out_ds = []
        
    def __len__(self):
        return len(self.idxs) #+ len(self.paths_LQ_ITS) + len(self.paths_HQ_LOL) + len(self.paths_HQ_Rain13K)
    
    
    def load_in(self, in_seq, seq_r, metadata, loaded_input=False):
        in0=time.time()
        input_data, in_masks = get_sequence(
        in_seq, 
        self.data_root, 
        self.data_split, 
        seq_r, 
        self.input_product,
        self.sat_bands, 
        self.preprocess_input, 
        self.swap_time_ch, 
        self.in_ds,
    )

        # print(np.shape(input_data), time.time()-in0,"in sequence time")
        return input_data, metadata

    def load_out(self, out_seq, seq_r, metadata):
        t1=time.time()
         #GROUND TRUTH (OUTPUT)
        if self.data_split not in ['test', 'heldout']:
            output_data, out_masks = get_sequence(out_seq, self.data_root, self.data_split, seq_r,
            self.output_product, [], self.preprocess_target,self.swap_time_ch, self.out_ds)

            # collapse time to channels
            metadata['target']['mask'] = out_masks
        else: #Just return [] if its test/heldout data
            output_data = np.array([])
        # print(time.time()-t1,"out sequence")
        return output_data, metadata

    def load_in_out(self, in_seq, out_seq=None, seq_r=None,):
        metadata = {'input': {'mask': [], 'timestamps': in_seq},
                    'target': {'mask': [], 'timestamps': out_seq},
                    'region': seq_r
                   }

        t0=time.time()
        input_data, metadata = self.load_in(in_seq, seq_r, metadata)
        output_data, metadata = self.load_out(out_seq, seq_r, metadata)

        # if self.sat_idx is not None:
        #     input_data = get_channels(input_data, self.sat_idx)

        if self.crop_edge is not None:
            input_data = crop_numpy(input_data, self.crop_edge, self.padding)
        

        # print(time.time()-t0,"seconds")
        return input_data, output_data, metadata
    
    def get_w4c_data(self, id):
            in_seq = self.idxs[id][0]
            out_seq = self.idxs[id][1]
            seq_r = self.idxs[id][2]
            sat, rad, metadata = self.load_in_out(in_seq, out_seq, seq_r)
            # print(sat.shape)
            sat = sat[[4, 10], :, :, :]
            return sat, rad, metadata


    def __getitem__(self, idx):

        dataset_choice = 'w4c'

        if self.split == 'training':
            if dataset_choice == 'w4c':
                sat, rad, metadata = self.get_w4c_data(idx)
                sat_in = einops.rearrange(sat, 'c l h w -> h w (c l)')
                rad_out = einops.rearrange(rad, 'c l h w -> h w (c l)')
                random_index = random.randint(0, len(self.idxs)-1)
                context_sat, context_rad, metadata = self.get_w4c_data(random_index)
                context_sat_in = einops.rearrange(context_sat, 'c l h w -> h w (c l)')
                context_rad_out = einops.rearrange(context_rad, 'c l h w -> h w (c l)')

                if self.phase == 'train':
                    # if the image size is too small
                    #print(HQ_size)
                    deg_type = 'w4c_forecast'
                    H, W, _ = sat_in.shape
                    #print(H, W)
                    if H != self.HQ_size or W != self.HQ_size:
                        sat_in = cv2.resize(np.copy(sat_in), (self.HQ_size, self.HQ_size),
                                            interpolation=cv2.INTER_LINEAR)
                        context_sat_in = cv2.resize(np.copy(context_sat_in), (self.HQ_size, self.HQ_size),
                                            interpolation=cv2.INTER_LINEAR)
                    # resized_image_data = cv2.resize(image_data, (224, 224))
                    H, W, _ = rad_out.shape
                    if H != self.HQ_size or W != self.HQ_size:
                        rad_out = np.expand_dims(cv2.resize(np.copy(rad_out), (self.HQ_size, self.HQ_size),
                                            interpolation=cv2.INTER_LINEAR), axis=2)

                        context_rad_out = np.expand_dims(cv2.resize(np.copy(context_rad_out), (self.HQ_size, self.HQ_size),
                                            interpolation=cv2.INTER_LINEAR), axis=2)

            if dataset_choice == 'w4c_reconstruct': 
                sat, rad, metadata = self.get_w4c_data(idx)

            img_HQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(sat_in, (2, 0, 1)))).float()
            img_HQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(rad_out, (2, 0, 1)))).float()
            img_LQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(context_sat_in, (2, 0, 1)))).float()
            img_LQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(context_rad_out, (2, 0, 1)))).float()

            input_query_img1 = img_LQ1
            target_img1 = img_LQ2
            input_query_img2 = img_HQ1
            target_img2 = img_HQ2
            batch = torch.stack([input_query_img1, target_img1, input_query_img2, target_img2], dim=0)
            # batch = [input_query_img1, target_img1, input_query_img2, target_img2]

            return batch, deg_type
        
        else:

            if dataset_choice == 'w4c':
                sat, rad, metadata = self.get_w4c_data(idx)
                sat_in = einops.rearrange(sat, 'c l h w -> h w (c l)')
                rad_out = einops.rearrange(rad, 'c l h w -> h w (c l)')
                random_index = random.randint(0, len(self.idxs)-1)
                context_sat, context_rad, metadata = self.get_w4c_data(random_index)
                context_sat_in = einops.rearrange(context_sat, 'c l h w -> h w (c l)')
                context_rad_out = einops.rearrange(context_rad, 'c l h w -> h w (c l)')

                if self.phase == 'train':
                    # if the image size is too small
                    #print(HQ_size)
                    deg_type = 'w4c_forecast'
                    H, W, _ = sat_in.shape
                    #print(H, W)
                    if H != self.HQ_size or W != self.HQ_size:
                        sat_in = cv2.resize(np.copy(sat_in), (self.HQ_size, self.HQ_size),
                                            interpolation=cv2.INTER_LINEAR)
                        context_sat_in = cv2.resize(np.copy(context_sat_in), (self.HQ_size, self.HQ_size),
                                            interpolation=cv2.INTER_LINEAR)
                    # resized_image_data = cv2.resize(image_data, (224, 224))
                    H, W, _ = rad_out.shape
                    if H != self.HQ_size or W != self.HQ_size:
                        rad_out = np.expand_dims(cv2.resize(np.copy(rad_out), (self.HQ_size, self.HQ_size),
                                            interpolation=cv2.INTER_LINEAR), axis=2)

                        context_rad_out = np.expand_dims(cv2.resize(np.copy(context_rad_out), (self.HQ_size, self.HQ_size),
                                            interpolation=cv2.INTER_LINEAR), axis=2)

            if dataset_choice == 'w4c_reconstruct': 
                sat, rad, metadata = self.get_w4c_data(idx)

            img_HQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(sat_in, (2, 0, 1)))).float()
            img_HQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(rad_out, (2, 0, 1)))).float()
            img_LQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(context_sat_in, (2, 0, 1)))).float()
            img_LQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(context_rad_out, (2, 0, 1)))).float()
            
            # batch = torch.stack([input_query_img1, target_img1, input_query_img2, target_img2], dim=0)
            # batch = [input_query_img1, target_img1, input_query_img2, target_img2]

            img_HQ2 = np.repeat(img_HQ2, 2, axis=0)
            img_LQ2 = np.repeat(img_LQ2, 2, axis=0)

            # return batch, deg_type
            input_query_img1 = img_HQ1
            target_img1 = img_HQ2
            input_query_img2 = img_LQ1
            target_img2 = img_LQ2

            # batch = torch.stack([input_query_img1, target_img1, input_query_img2, target_img2], dim=0)
            
            batch = {'input_query_img1': input_query_img1, 'target_img1': target_img1,
                    'input_query_img2': input_query_img2, 'target_img2': target_img2}
            
            return batch, deg_type


class DatasetSevir_Train(Dataset):
    def __init__(self, data_path, input_size, phase, task='sevir'):
        
        np.random.seed(5)
        self.HQ_size = input_size
        self.phase = phase
        self.task = task
        self.data_choice = data_path
        # base dataset
        # self.paths_HQ, self.sizes_HQ = util.get_image_paths('img', dataset_path)
        # self.paths_base_len = len(self.paths_HQ)
        # print('dataset for test: ', len(self.paths_HQ))
        self.catalog = pd.read_csv('dataset/output.csv', low_memory=False)
        img_types = set(['vis','ir069','ir107','vil'])
        events = self.catalog.groupby('id').filter(lambda x: img_types.issubset(set(x['img_type']))).groupby('id')
        self.event_ids = list(events.groups.keys())

        self.dataset_list = ['sevir', 'trans', 'inter']

        if self.phase == 'train':
            # self.event_ids = self.event_ids[:11408]
            self.event_ids = self.event_ids[:11458]
        else:
            self.event_ids = self.event_ids[11488:]
    
    def get_line_time(self, value):
        row = self.catalog.loc[self.catalog['id'] == value].iloc[0]
        time = row['file_name'].split('/')[1]
        return time
    
    def is_day_time(self, value):
        row = self.catalog.loc[self.catalog['id'] == value].iloc[0]
        time = row['time_utc'].split(' ')[1]
        return time

    def __len__(self):
        if self.task == 'sevir':
            if self.phase == 'train':
                # return len(self.event_ids)*49
                return len(self.event_ids)*24
            else:
                return len(self.event_ids)*4
        elif self.task == 'trans':
            # return len(self.event_ids)*49
            if self.phase == 'train':
                return len(self.event_ids)*24
            else:
                return len(self.event_ids)*4
        elif self.task == 'inter':
            if self.phase == 'train':
                return len(self.event_ids)*36
            else:
                return len(self.event_ids)*4
        elif self.task == 'predict':
            if self.phase == 'train':
                return len(self.event_ids)*24
            else:
                return len(self.event_ids)*4
        elif self.task == 'ir_predict':
            if self.phase == 'train':
                return len(self.event_ids)*24
            else:
                return len(self.event_ids)*4
        elif self.task == '2c_tasks':
            if self.phase == 'train':
                return len(self.event_ids)*24
            else:
                return len(self.event_ids)*4
        elif self.task == 'down_scaling':
            if self.phase == 'train':
                return len(self.event_ids)*24
            else:
                return len(self.event_ids)*4
        elif self.task == 'vis_recon':
            if self.phase == 'train':
                return len(self.event_ids)*24
            else:
                return len(self.event_ids)*4
    
    def get_sevir_in_out(self, ids):
        # evrty event has 49 frame

        if self.phase == 'train':
            event_id = ids // 24
            frame_id = (ids % 24)*2
        else:
            event_id = ids // 4
            frame_id = (ids % 4)*10

        event = self.event_ids[event_id]
        event_time = self.get_line_time(event)

        in_variables = ['ir069', 'ir107']
        # out_variables = ['vis']
        out_variables = ['vil']
        event_data = []
        out_data = []

        for variable in in_variables:
            path = f"s3://sevir_pair/{variable}/{event_time}/{event}.npy"
            data = io.BytesIO(client.get(path))
            data = np.load(data)
            data = data[:, :, frame_id]
            # normalization
            if variable == 'ir069':
                data = (data-zscore_normalizations_sevir['ir069']['shift'])/zscore_normalizations_sevir['ir069']['scale']
            else:
                data = (data-zscore_normalizations_sevir['ir107']['shift'])/zscore_normalizations_sevir['ir107']['scale']
            event_data.append(data)
        sat_in = np.stack(event_data, axis=2)

        for variable in out_variables:
            path = f"s3://sevir_pair/{variable}/{event_time}/{event}.npy"
            data = io.BytesIO(client.get(path))
            data = np.load(data)
            data = data[:, :, frame_id]
            # normalization
            if variable == 'vil':
                data = (data-zscore_normalizations_sevir['vil']['shift'])/zscore_normalizations_sevir['vil']['scale']
                data = np.expand_dims(data, axis=-1)
            else:
                print('only can use vil as target!')
            out_data.append(data)
        # sat_out = np.repeat(out_data, 2, axis=0)
        # sat_out = np.transpose(sat_out, (1, 2, 0))
        sat_out = out_data[0]
        return sat_in, sat_out
    
    def get_random_day_vis(self, ids):
        if self.phase == 'train':
            event_id = ids // 24
            frame_id = (ids % 24)*2
        else:
            event_id = ids // 4
            frame_id = (ids % 4)*10
        event = self.event_ids[event_id]
        event_time = self.get_line_time(event)
        path = f"s3://sevir_pair/vis/{event_time}/{event}.npy"
        data = io.BytesIO(client.get(path))
        data = np.load(data)
        data = data[:, :, frame_id]
        if data.mean() > 100:
            return event_id, frame_id
        else:
            if self.phase == 'train':
                random_ids = random.randint(0, len(self.event_ids)*24-1)
            else:
                random_ids = random.randint(0, len(self.event_ids)*4-1)
            return self.get_random_day_vis(random_ids)

    def get_vis_reconstruction(self, ids):
        # evrty event has 49 frame

        event_id, frame_id = self.get_random_day_vis(ids)

        event = self.event_ids[event_id]
        event_time = self.get_line_time(event)

        in_variables = ['ir069', 'ir107']
        # out_variables = ['vis']
        out_variables = ['vis']
        event_data = []
        out_data = []

        for variable in out_variables:
            path = f"s3://sevir_pair/{variable}/{event_time}/{event}.npy"
            data = io.BytesIO(client.get(path))
            data = np.load(data)
            data = data[:, :, frame_id]
            # normalization
            data = (data-zscore_normalizations_sevir['vis']['shift'])/zscore_normalizations_sevir['vis']['scale']
            data = np.expand_dims(data, axis=-1)
            out_data.append(data)
        sat_out = out_data[0]

        for variable in in_variables:
            path = f"s3://sevir_pair/{variable}/{event_time}/{event}.npy"
            data = io.BytesIO(client.get(path))
            data = np.load(data)
            data = data[:, :, frame_id]
            # normalization
            if variable == 'ir069':
                data = (data-zscore_normalizations_sevir['ir069']['shift'])/zscore_normalizations_sevir['ir069']['scale']
            else:
                data = (data-zscore_normalizations_sevir['ir107']['shift'])/zscore_normalizations_sevir['ir107']['scale']
            event_data.append(data)
        sat_in = np.stack(event_data, axis=2)

        return sat_in, sat_out
    
    def get_ir_trans(self, ids):
        # evrty event has 49 frame
        if self.phase == 'train':
            event_id = ids // 24
            frame_id = (ids % 24)*2

        else:
            event_id = ids // 4
            frame_id = (ids % 4)*10

        event = self.event_ids[event_id]
        event_time = self.get_line_time(event)

        in_variables = ['ir069']
        # out_variables = ['vis']
        out_variables = ['ir107']
        event_data = []
        out_data = []

        for variable in in_variables:
            path = f"s3://sevir_pair/{variable}/{event_time}/{event}.npy"
            data = io.BytesIO(client.get(path))
            data = np.load(data)
            data = data[:, :, frame_id]
            # normalization
            if variable == 'ir069':
                data = (data-zscore_normalizations_sevir['ir069']['shift'])/zscore_normalizations_sevir['ir069']['scale']
                data = np.dstack((data, data))
            else:
                data = (data-zscore_normalizations_sevir['ir107']['shift'])/zscore_normalizations_sevir['ir107']['scale']
                data = np.dstack((data, data))
            event_data.append(data)
        sat_in = event_data[0]

        for variable in out_variables:
            path = f"s3://sevir_pair/{variable}/{event_time}/{event}.npy"
            data = io.BytesIO(client.get(path))
            data = np.load(data)
            data = data[:, :, frame_id]
            # normalization
            if variable == 'ir107':
                data = (data-zscore_normalizations_sevir['ir107']['shift'])/zscore_normalizations_sevir['ir107']['scale']
                data = np.expand_dims(data, axis=-1)
            else:
                data = (data-zscore_normalizations_sevir['ir069']['shift'])/zscore_normalizations_sevir['ir069']['scale']
                data = np.expand_dims(data, axis=-1)
            out_data.append(data)
        # sat_out = np.repeat(out_data, 2, axis=0)
        # sat_out = np.transpose(sat_out, (1, 2, 0))
        sat_out = out_data[0]
        return sat_in, sat_out
    
    def get_sevir_predict(self, ids):
        # evrty event has 49 frame
        if self.phase == 'train':
            event_id = ids // 24
            frame_id = ids % 24

        else:
            event_id = ids // 4
            frame_id = (ids % 4)*5
            
        event = self.event_ids[event_id]
        event_time = self.get_line_time(event)

        in_variables = ['vil']
        data_list = [frame_id, frame_id+6, frame_id+12, frame_id+18, frame_id+24]

        for variable in in_variables:
            path = f"s3://sevir_pair/{variable}/{event_time}/{event}.npy"
            data = io.BytesIO(client.get(path))
            data = np.load(data)
            data = data[:, :, data_list]
            # normalization
            if variable == 'vil':
                data = (data-zscore_normalizations_sevir['vil']['shift'])/zscore_normalizations_sevir['vil']['scale']
                # data = np.expand_dims(data, axis=-1)
        vil_in = data[:,:,:-1]
        vil_out = data[:,:,-1]
        vil_out = np.expand_dims(vil_out, axis=-1)
        return vil_in, vil_out
    
    def get_ir_predict(self, ids, variable_type=None):
        # evrty event has 49 frame
        if self.phase == 'train':
            event_id = ids // 24
            frame_id = ids % 24

        else:
            event_id = ids // 4
            frame_id = (ids % 4)*5
            
        event = self.event_ids[event_id]
        event_time = self.get_line_time(event)

        in_variables = ['ir069']
        data_list = [frame_id, frame_id+6, frame_id+12, frame_id+18, frame_id+24]

        for variable in in_variables:
            path = f"s3://sevir_pair/{variable}/{event_time}/{event}.npy"
            data = io.BytesIO(client.get(path))
            data = np.load(data)
            data = data[:, :, data_list]
            # normalization
            if variable == 'ir069':
                data = (data-zscore_normalizations_sevir['ir069']['shift'])/zscore_normalizations_sevir['ir069']['scale']
                # data = np.expand_dims(data, axis=-1)
        ir_in = data[:,:,:-1]
        ir_out = data[:,:,-1]
        ir_out = np.expand_dims(ir_out, axis=-1)
        return ir_in, ir_out
    
    def get_sevir_inter(self, ids):
        # evrty event has 49 frame
        if self.phase == 'train':
            event_id = ids // 36
            frame_id = ids % 36
        else:
            event_id = ids // 4
            frame_id = (ids % 4)*10

        event = self.event_ids[event_id]
        event_time = self.get_line_time(event)

        in_variables = ['vil']
        data_list = [frame_id, frame_id+12, frame_id+6]

        for variable in in_variables:
            path = f"s3://sevir_pair/{variable}/{event_time}/{event}.npy"
            data = io.BytesIO(client.get(path))
            data = np.load(data)
            data = data[:, :, data_list]
            # normalization
            if variable == 'vil':
                data = (data-zscore_normalizations_sevir['vil']['shift'])/zscore_normalizations_sevir['vil']['scale']
                # data = np.expand_dims(data, axis=-1)
        vil_in = data[:,:,:-1]
        vil_out = data[:,:,-1]
        vil_out = np.expand_dims(vil_out, axis=-1)
        return vil_in, vil_out
    
    def get_sevir_vil_down_scaling(self, ids, variable_type=['vil']):
        # evrty event has 49 frame
        if self.phase == 'train':
            event_id = ids // 24
            frame_id = (ids % 24)*2
        else:
            event_id = ids // 4
            frame_id = (ids % 4)*10

        event = self.event_ids[event_id]
        event_time = self.get_line_time(event)

        in_variables = variable_type

        for variable in in_variables:
            path = f"s3://sevir_pair/{variable}/{event_time}/{event}.npy"
            data = io.BytesIO(client.get(path))
            data = np.load(data)
            data = data[:, :, frame_id]
            # normalization
            if variable == 'vil':
                data = (data-zscore_normalizations_sevir['vil']['shift'])/zscore_normalizations_sevir['vil']['scale']
                # data = np.expand_dims(data, axis=-1)
            if variable == 'ir069':
                data = (data-zscore_normalizations_sevir['ir069']['shift'])/zscore_normalizations_sevir['ir069']['scale']
        vil_in = data[:,:]
        vil_out = data[:,:]
        vil_in = np.expand_dims(vil_in, axis=-1)
        vil_in = np.dstack((vil_in, vil_in))
        vil_out = np.expand_dims(vil_out, axis=-1)
        if variable_type[0] == 'vil':
            # down_scaling: vil_in 384->256->64
            vil_in = cv2.resize(np.copy(vil_in), (64, 64),
                        interpolation=cv2.INTER_LINEAR)
        else:
            vil_in = cv2.resize(np.copy(vil_in), (64, 64),
                        interpolation=cv2.INTER_LINEAR)
        return vil_in, vil_out

    def __getitem__(self, idx):

        if self.task == '2c_tasks':
            dataset_choice = np.random.choice(self.dataset_list, p=[1/3, 1/3, 1/3])
            # dataset_choice = self.data_choice
        else:
            dataset_choice = self.data_choice
        deg_type = dataset_choice
        #todo event_ids并未对齐全部数据集
        random_index = random.randint(0, len(self.event_ids)-1)
        if dataset_choice == 'sevir':
            sat_in, rad_out = self.get_sevir_in_out(idx)
            context_sat_in, context_rad_out = self.get_sevir_in_out(random_index)
        elif dataset_choice == 'ir_trans':
            sat_in, rad_out = self.get_ir_trans(idx)
            context_sat_in, context_rad_out = self.get_ir_trans(random_index)
        elif dataset_choice == 'predict':
            sat_in, rad_out = self.get_sevir_predict(idx)
            context_sat_in, context_rad_out = self.get_sevir_predict(random_index)
        elif dataset_choice == 'inter':
            sat_in, rad_out = self.get_sevir_inter(idx)
            context_sat_in, context_rad_out = self.get_sevir_inter(random_index)
        elif dataset_choice == 'down_scaling_vil':
            sat_in, rad_out = self.get_sevir_vil_down_scaling(idx, variable_type=['vil'])
            context_sat_in, context_rad_out = self.get_sevir_vil_down_scaling(random_index, variable_type=['vil'])
        elif dataset_choice == 'down_scaling_ir':
            sat_in, rad_out = self.get_sevir_vil_down_scaling(idx, variable_type=['ir069'])
            context_sat_in, context_rad_out = self.get_sevir_vil_down_scaling(random_index, variable_type=['ir069'])
        elif dataset_choice == 'ir_predict':
            sat_in, rad_out = self.get_ir_predict(idx, variable_type=['ir069'])
            context_sat_in, context_rad_out = self.get_ir_predict(random_index, variable_type=['ir069'])
        elif dataset_choice == 'vis_recon':
            sat_in, rad_out = self.get_vis_reconstruction(idx)
            context_sat_in, context_rad_out = self.get_vis_reconstruction(random_index)
        H, W, _ = sat_in.shape
        if H != self.HQ_size or W != self.HQ_size:
            sat_in = cv2.resize(np.copy(sat_in), (self.HQ_size, self.HQ_size),
                                interpolation=cv2.INTER_LINEAR)
            context_sat_in = cv2.resize(np.copy(context_sat_in), (self.HQ_size, self.HQ_size),
                                interpolation=cv2.INTER_LINEAR)
        H, W, _ = rad_out.shape
        if H != self.HQ_size or W != self.HQ_size:
            rad_out = np.expand_dims(cv2.resize(np.copy(rad_out), (self.HQ_size, self.HQ_size),
                                interpolation=cv2.INTER_LINEAR), axis=2)

            context_rad_out = np.expand_dims(cv2.resize(np.copy(context_rad_out), (self.HQ_size, self.HQ_size),
                                interpolation=cv2.INTER_LINEAR), axis=2)

        if self.phase == 'train':
            # hwc->chw
            img_HQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(sat_in, (2, 0, 1)))).float()
            img_HQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(rad_out, (2, 0, 1)))).float()
            img_LQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(context_sat_in, (2, 0, 1)))).float()
            img_LQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(context_rad_out, (2, 0, 1)))).float()
            
            img_HQ2 = np.repeat(img_HQ2, 2, axis=0)
            img_LQ2 = np.repeat(img_LQ2, 2, axis=0)

            input_query_img1 = img_LQ1
            target_img1 = img_LQ2
            input_query_img2 = img_HQ1
            target_img2 = img_HQ2

            # if dataset_choice == 'predict':
            #     batch = [input_query_img1, target_img1, input_query_img2, target_img2]
            # else:
            #     batch = torch.stack([input_query_img1, target_img1, input_query_img2, target_img2], dim=0)
            batch = [input_query_img1, target_img1, input_query_img2, target_img2]

            return batch, deg_type
        
        else:
            img_HQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(sat_in, (2, 0, 1)))).float()
            img_HQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(rad_out, (2, 0, 1)))).float()
            img_LQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(context_sat_in, (2, 0, 1)))).float()
            img_LQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(context_rad_out, (2, 0, 1)))).float()
            
            img_HQ2 = np.repeat(img_HQ2, 2, axis=0)
            img_LQ2 = np.repeat(img_LQ2, 2, axis=0)

            input_query_img1 = img_LQ1
            target_img1 = img_LQ2
            input_query_img2 = img_HQ1
            target_img2 = img_HQ2

            # batch = torch.stack([input_query_img1, target_img1, input_query_img2, target_img2], dim=0)
            
            batch = {'input_query_img1': input_query_img1, 'target_img1': target_img1,
                    'input_query_img2': input_query_img2, 'target_img2': target_img2}
            
            return batch, deg_type

            # return batch, deg_type

class Dataset_dblur(Dataset):
    def __init__(self, split='train'):
        super().__init__()
        self.height = 256
        self.width = 256
        self.ceph_root = 's3://gongjunchao/radar_deblur/blur_data/I10O12/sevir/incepu/TimeStep12'
        self.file_list = self._init_file_list(split)
        self.split = split
        
        ## sproject client ##
        self.client = Client("~/petreloss.conf")

        assert len(self.file_list) == self.__len__()
        print('SEVIR {} number: {}'.format(str(self.split), len(self.file_list)))

    def _init_file_list(self, split):
        """
        preprocessed total length: 12120
        """
        if split == 'train':
            files = list(range(0, 10500))
        elif split == 'valid':
            files = list(range(10500, 12000))
        elif split == 'test':
            files = list(range(10500, 12000))
        return files
    
    def __len__(self):
        return len(self.file_list)

    def _load_frames(self, file, type):
        file_path = os.path.join(self.ceph_root, f'{type}_{file}.npy')
        with io.BytesIO(self.client.get(file_path)) as f:
            frame_data = np.load(f)
        tensor = torch.from_numpy(frame_data)
        return tensor

    def _get_dataset_name(self):
        dataset_name = 'sevir'
        return dataset_name 
    
    # def _resize(self, frame_data):
    #     _, _, H, W = frame_data.shape 
    #     if H != self.height or W != self.height:
    #         frame_data = nn.functional.interpolate(frame_data, size=(self.height, self.width), mode='bilinear')
    #     return frame_data
    
    def __getitem__(self, index):
        file = self.file_list[index]
        blur_frame_data = self._load_frames(file, type='pred')
        gt_frame_data = self._load_frames(file, type='tar')
        random_index = random.randint(0, len(self.file_list)-1)
        random_file = self.file_list[random_index]
        context_blur_frame_data = self._load_frames(random_file, type='pred')
        context_gt_frame_data = self._load_frames(random_file, type='tar')

        packed_results = dict()
        packed_results['inputs'] = blur_frame_data
        packed_results['data_samples'] = gt_frame_data
        packed_results['dataset_name'] = self._get_dataset_name()
        packed_results['file_name'] = index

        deg_type='dblur'

        if self.split == 'train':
            img_HQ1 = torch.from_numpy(np.ascontiguousarray(blur_frame_data)).float()
            img_HQ2 = torch.from_numpy(np.ascontiguousarray(gt_frame_data)).float()
            img_LQ1 = torch.from_numpy(np.ascontiguousarray(context_blur_frame_data)).float()
            img_LQ2 = torch.from_numpy(np.ascontiguousarray(context_gt_frame_data)).float()

            img_HQ1 = np.repeat(img_HQ1, 2, axis=0)
            img_LQ1 = np.repeat(img_LQ1, 2, axis=0)
            img_HQ2 = np.repeat(img_HQ2, 2, axis=0)
            img_LQ2 = np.repeat(img_LQ2, 2, axis=0)

            input_query_img1 = img_LQ1
            target_img1 = img_LQ2
            input_query_img2 = img_HQ1
            target_img2 = img_HQ2

            # if dataset_choice == 'predict':
            #     batch = [input_query_img1, target_img1, input_query_img2, target_img2]
            # else:
            #     batch = torch.stack([input_query_img1, target_img1, input_query_img2, target_img2], dim=0)
            batch = [input_query_img1, target_img1, input_query_img2, target_img2]

            return batch, deg_type
        
        else:
            img_HQ1 = torch.from_numpy(np.ascontiguousarray(blur_frame_data)).float()
            img_HQ2 = torch.from_numpy(np.ascontiguousarray(gt_frame_data)).float()
            img_LQ1 = torch.from_numpy(np.ascontiguousarray(context_blur_frame_data)).float()
            img_LQ2 = torch.from_numpy(np.ascontiguousarray(context_gt_frame_data)).float()

            img_HQ1 = np.repeat(img_HQ1, 2, axis=0)
            img_LQ1 = np.repeat(img_LQ1, 2, axis=0)
            img_HQ2 = np.repeat(img_HQ2, 2, axis=0)
            img_LQ2 = np.repeat(img_LQ2, 2, axis=0)

            input_query_img1 = img_LQ1
            target_img1 = img_LQ2
            input_query_img2 = img_HQ1
            target_img2 = img_HQ2

            # batch = torch.stack([input_query_img1, target_img1, input_query_img2, target_img2], dim=0)
            
            batch = {'input_query_img1': input_query_img1, 'target_img1': target_img1,
                    'input_query_img2': input_query_img2, 'target_img2': target_img2}
            
            return batch, deg_type


class DatasetLowlevel_Val(Dataset):
    def __init__(self, dataset_path, input_size=320):
        self.paths_HQ, self.sizes_HQ = util.get_image_paths('img', dataset_path)
        self.input_size = input_size

    def __len__(self):
        return len(self.paths_HQ)-1


    def __getitem__(self, idx):
        HQ1_path = self.paths_HQ[idx]
        random_index = random.randint(0, len(self.paths_HQ)-1)
        HQ2_path = self.paths_HQ[random_index]
        
        img_HQ1 = util.read_img(None, HQ1_path, None)
        img_HQ2 = util.read_img(None, HQ2_path, None)
        
        #img_HQ1 = util.modcrop(img_HQ1, 16)
        #img_HQ2 = util.modcrop(img_HQ2, 16)
        
        img_HQ1 = cv2.resize(np.copy(img_HQ1), (self.input_size, self.input_size),
                                    interpolation=cv2.INTER_LINEAR)
        img_HQ2 = cv2.resize(np.copy(img_HQ2), (self.input_size, self.input_size),
                                    interpolation=cv2.INTER_LINEAR)
        
        if img_HQ1.ndim == 2:
            img_HQ1 = np.expand_dims(img_HQ1, axis=2)
            img_HQ1 = np.concatenate((img_HQ1, img_HQ1, img_HQ1), axis=2)
        if img_HQ2.ndim == 2:
            img_HQ2 = np.expand_dims(img_HQ2, axis=2)
            img_HQ2 = np.concatenate((img_HQ2, img_HQ2, img_HQ2), axis=2)
            
        if img_HQ1.shape[2] !=3:
            img_HQ1 = np.concatenate((img_HQ1, img_HQ1, img_HQ1), axis=2)
        if img_HQ2.shape[2] !=3:
            img_HQ2 = np.concatenate((img_HQ2, img_HQ2, img_HQ2), axis=2)
        
        # add degradation to HQ
        degradation_type_list = ['GaussianNoise', 'GaussianBlur', 'JPEG', 'Resize', 
                                    'Rain', 'SPNoise', 'LowLight', 'PoissonNoise', 'Ringing',
                                    'r_l', 'Inpainting', 'Laplacian', 'Canny']
        
        degradation_type_1 = ['Resize', 'LowLight', '']
        degradation_type_2 = ['GaussianBlur', '']
        degradation_type_3 = ['GaussianNoise', 'SPNoise', 'PoissonNoise', '']
        degradation_type_4 = ['JPEG', 'Ringing', 'r_l', '']
        degradation_type_5 = ['Inpainting', '']
        degradation_type_6 = ['Rain', '']
        
        round_select = np.random.choice(['1', 'Single'], p=[4/5, 1/5])
            
        if round_select == '1':
            # 1 round
            deg_type1 = random.choice(degradation_type_1)
            deg_type2 = random.choice(degradation_type_2)
            deg_type3 = random.choice(degradation_type_3)
            deg_type4 = random.choice(degradation_type_4)
            deg_type5 = random.choice(degradation_type_5)
            deg_type6 = random.choice(degradation_type_6)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(np.copy(img_HQ1), np.copy(img_HQ2), deg_type1)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(np.copy(img_LQ1), np.copy(img_LQ2), deg_type2)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(np.copy(img_LQ1), np.copy(img_LQ2), deg_type3)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(np.copy(img_LQ1), np.copy(img_LQ2), deg_type4)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(np.copy(img_LQ1), np.copy(img_LQ2), deg_type5)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(np.copy(img_LQ1), np.copy(img_LQ2), deg_type6)
            deg_type = 'R1_'+deg_type1+'_'+deg_type2+'_'+deg_type3+'_'+deg_type4+'_'+deg_type5+'_'+deg_type6
            # print(deg_type)
            # print(img_HQ1.shape)
            # print(img_LQ1.shape)
            # print(img_HQ2.shape)
            # print(img_LQ2.shape)
        elif round_select == 'Single':
            deg_type1 = random.choice(degradation_type_list)
            img_LQ1, img_LQ2, img_HQ1, img_HQ2 = add_degradation_two_images(img_HQ1, img_HQ2, deg_type1)
            deg_type = deg_type1
            
        elif round_select == '2':
            # 2 round (currently does not take into account, 1 round is enough)
            deg_type1 = random.choice(degradation_type_1)
            deg_type2 = random.choice(degradation_type_2)
            deg_type3 = random.choice(degradation_type_3)
            deg_type4 = random.choice(degradation_type_4)
            deg_type5 = random.choice(degradation_type_5)
            deg_type6 = random.choice(degradation_type_6)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(img_HQ1, img_HQ2, deg_type1)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(img_LQ1, img_LQ2, deg_type2)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(img_LQ1, img_LQ2, deg_type3)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(img_LQ1, img_LQ2, deg_type4)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(img_LQ1, img_LQ2, deg_type5)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(img_LQ1, img_LQ2, deg_type6)
            
            deg_type7 = random.choice(degradation_type_1)
            deg_type8 = random.choice(degradation_type_2)
            deg_type9 = random.choice(degradation_type_3)
            deg_type10 = random.choice(degradation_type_4)
            deg_type11 = random.choice(degradation_type_5)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(img_LQ1, img_LQ2, deg_type7)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(img_LQ1, img_LQ2, deg_type8)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(img_LQ1, img_LQ2, deg_type9)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(img_LQ1, img_LQ2, deg_type10)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(img_LQ1, img_LQ2, deg_type11)
            
            deg_type = 'R2_'+deg_type1+'_'+deg_type2+'_'+deg_type3+'_'+deg_type4+'_'+deg_type5+'_'+deg_type6+'_'+ \
                        deg_type7+'_'+deg_type8+'_'+deg_type9+'_'+deg_type10+'_'+deg_type11

       
        
        
        # print(HQ1_path)
        # print(HQ2_path) 
        # print(deg_type)
        # print(img_HQ1.shape)
        # print(img_LQ1.shape)
        # print(img_HQ2.shape)
        # print(img_LQ2.shape)
        
        # BGR to RGB, HWC to CHW, numpy to tensor
        # if img_HQ1.shape[2] == 3:
        #     img_HQ1 = img_HQ1[:, :, [2, 1, 0]]
        #     img_HQ2 = img_HQ2[:, :, [2, 1, 0]]
        #     img_LQ1 = img_LQ1[:, :, [2, 1, 0]]
        #     img_LQ2 = img_LQ2[:, :, [2, 1, 0]]
        
        img_HQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ1, (2, 0, 1)))).float()
        img_HQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ2, (2, 0, 1)))).float()
        img_LQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ1, (2, 0, 1)))).float()
        img_LQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ2, (2, 0, 1)))).float()
        
        
        input_query_img1 = img_LQ1
        target_img1 = img_HQ1
        input_query_img2 = img_LQ2
        target_img2 = img_HQ2
        
        batch = {'input_query_img1': input_query_img1, 'target_img1': target_img1,
                 'input_query_img2': input_query_img2, 'target_img2': target_img2}
        return batch, deg_type


    def __init__(self, dataset_path_root):
        self.dataset_path_root = dataset_path_root
        self.dataset_paths = os.listdir(self.dataset_path_root)
        sorted(self.dataset_paths)

    def __len__(self):
        return len(self.dataset_paths)


    def __getitem__(self, idx):
        LQ1_path = os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'prompt_input_img1.png')
        HQ1_path = os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'prompt_target_img1.png')
        LQ2_path = os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'query_input_img2.png')
        if os.path.exists(os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'query_target_img2.png')):
            HQ2_path = os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'query_target_img2.png')
        else:
            HQ2_path = None
        
        img_HQ1 = util.read_img(None, HQ1_path, None)
        img_LQ1 = util.read_img(None, LQ1_path, None)
        
        if HQ2_path:
            img_HQ2 = util.read_img(None, HQ2_path, None)
        
        img_LQ2 = util.read_img(None, LQ2_path, None)
        
        
        
        if img_HQ1.ndim == 2:
            img_HQ1 = np.expand_dims(img_HQ1, axis=2)
            img_HQ1 = np.concatenate((img_HQ1, img_HQ1, img_HQ1), axis=2)
        if img_LQ1.ndim == 2:
            img_LQ1 = np.expand_dims(img_LQ1, axis=2)
            img_LQ1 = np.concatenate((img_LQ1, img_LQ1, img_LQ1), axis=2)
        if HQ2_path and img_HQ2.ndim == 2:
            img_HQ2 = np.expand_dims(img_HQ2, axis=2)
            img_HQ2 = np.concatenate((img_HQ2, img_HQ2, img_HQ2), axis=2)
        if img_LQ2.ndim == 2:
            img_LQ2 = np.expand_dims(img_LQ2, axis=2)
            img_LQ2 = np.concatenate((img_LQ2, img_LQ2, img_LQ2), axis=2)
            
        if img_HQ1.shape[2] !=3:
            img_HQ1 = np.concatenate((img_HQ1, img_HQ1, img_HQ1), axis=2)
        if img_LQ1.shape[2] !=3:
            img_LQ1 = np.concatenate((img_LQ1, img_LQ1, img_LQ1), axis=2)
        if HQ2_path and img_HQ2.shape[2] !=3:
            img_HQ2 = np.concatenate((img_HQ2, img_HQ2, img_HQ2), axis=2)
        if img_LQ2.shape[2] !=3:
            img_LQ2 = np.concatenate((img_LQ2, img_LQ2, img_LQ2), axis=2)
        
        
        # img_HQ1 = cv2.resize(np.copy(img_HQ1), (112, 112),
        #                             interpolation=cv2.INTER_LINEAR)
        # img_LQ1 = cv2.resize(np.copy(img_LQ1), (112, 112),
        #                             interpolation=cv2.INTER_LINEAR)
        # if HQ2_path:
        #     img_HQ2 = cv2.resize(np.copy(img_HQ2), (112, 112),
        #                             interpolation=cv2.INTER_LINEAR)
        # img_LQ2 = cv2.resize(np.copy(img_LQ2), (112, 112),
        #                             interpolation=cv2.INTER_LINEAR)
        
        img_HQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ1, (2, 0, 1)))).float()
        if HQ2_path:
            img_HQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ2, (2, 0, 1)))).float()
        img_LQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ1, (2, 0, 1)))).float()
        img_LQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ2, (2, 0, 1)))).float()
        
        
        input_query_img1 = img_LQ1
        target_img1 = img_HQ1
        input_query_img2 = img_LQ2
        if HQ2_path:
            target_img2 = img_HQ2
        else:
            target_img2 = 'None'
        
        batch = {'input_query_img1': input_query_img1, 'target_img1': target_img1,
                 'input_query_img2': input_query_img2, 'target_img2': target_img2,
                 'input_query_img2_path': LQ2_path}
        
        deg_type = self.dataset_paths[idx]
        
        return batch, deg_type