from torch.utils.data import Dataset
from petrel_client.client import Client
from tqdm import tqdm
import numpy as np
import io
import time
import xarray as xr
import json
import pandas as pd
import os
import gc
from multiprocessing import Pool
from multiprocessing import shared_memory
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import copy
import queue
import torch
import cv2
import random
# from s3_client import s3_client

Years = {
    'train': ['1979-01-01 00:00:00', '2016-12-31 23:00:00'],
    'valid': ['2018-01-01 00:00:00', '2018-12-31 23:00:00'],
    'test': ['2016-01-01 00:00:00', '2017-12-31 23:00:00'],
    'all': ['1979-01-01 00:00:00', '2020-12-31 23:00:00']
}

multi_level_vnames = [
    "z", "t", "q", "r", "u", "v"
]
single_level_vnames = [
    "t2m", "u10", "v10"
]
long2shortname_dict = {"geopotential": "z", "temperature": "t", "specific_humidity": "q", "relative_humidity": "r", "u_component_of_wind": "u", "v_component_of_wind": "v", "vorticity": "vo", "potential_vorticity": "pv", \
    "2m_temperature": "t2m", "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10", "total_cloud_cover": "tcc", "total_precipitation": "tp", "toa_incident_solar_radiation": "tisr"}
constants = [
    "lsm", "orography", "slt"
]
height_level = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, \
    500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]

multi_level_dict_param = {"z":height_level, "t": height_level, "q": height_level, "r": height_level}


def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma



class era5_128x256(Dataset):
    def __init__(self, data_dir='s3://era5_np128x256', split='train', out_var=8, lead_time=None,**kwargs) -> None:
        super().__init__()
        # print("init begin")
        self.length = kwargs.get('length', 1)
        self.file_stride = kwargs.get('file_stride', 6)
        self.sample_stride = kwargs.get('sample_stride', 1)
        self.output_meanstd = kwargs.get("output_meanstd", False)
        self.pred_length = kwargs.get("pred_length", 1)
        self.use_diff_pos = kwargs.get("use_diff_pos", False)
        # self.rm_equator = kwargs.get("rm_equator", True)
        Years_dict = kwargs.get('years', Years)

        vnames_type = kwargs.get("vnames", {})
        self.constants_types = vnames_type.get('constants', ["lsm", "orography", "slt"])
        self.single_level_vnames = vnames_type.get('single_level_vnames', ['t2m','u10','v10'])
        self.multi_level_vnames = vnames_type.get('multi_level_vnames', ['z','u', 'v', 't', 'r', 'q'])
        self.height_level_list = vnames_type.get('hight_level_list', [50, 250, 500, 600, 700, 850, 925])
        self.height_level_indexes = [height_level.index(j) for j in self.height_level_list]
        # multi_level_dict = vnames_type.get('multi_level_vnames_dict', {"z": height_level, "v": height_level})
        # multi_height_dict = vnames_type.get("height_level", None)
        # self.s3_client = s3_client(bucket_name='era5_nc')

        # constants_index = [constants.index(constant) for constant in self.constants_types]
        # self.select_row = [i for i in range(721)]
        # if self.rm_equator:
        #     del self.select_row[360]
        self.split = split
        self.out_var = out_var
        self.lead_time = lead_time
        self.data_dir = data_dir
        self.client = Client(conf_path="~/petreloss.conf")
        # self.client = Client(conf_path="~/petreloss.conf")
        years = Years_dict[split]
        self.init_file_list(years)

        # print(constants_index)
        if len(self.constants_types) > 0:
            self.constants_data = self.get_constants_data(self.constants_types)
        else:
            self.constants_data = None
        self._get_meanstd()

        self.data_element_num = len(self.single_level_vnames) + len(self.multi_level_vnames) * len(self.height_level_list)
        dim = len(self.single_level_vnames) + len(self.multi_level_vnames) * len(self.height_level_list)

        # self.index_dict1 = {"u10": 0, "v10": 1, "t2m": 2, 'msl': 3, 'z': 4, 'q': 41, 'u': 78, 'v': 115, 't': 152, 'w': 189}
        # self.index_dict2 = {"u10": 1, "v10": 2, "t2m": 3, 'msl': 4, 'z': 41, 'q': 78, 'u': 115, 'v': 152, 't': 189, 'w': 226}


        self.index_dict1 = {}
        self.index_dict2 = {}
        i = 0
        for vname in self.single_level_vnames:
            self.index_dict1[(vname, 0)] = i
            i += 1
        for vname in self.multi_level_vnames:
            for height in self.height_level_list:
                self.index_dict1[(vname, height)] = i
                i += 1

        self.index_queue = multiprocessing.Queue()
        self.unit_data_queue = multiprocessing.Queue()

        self.index_queue.cancel_join_thread() 
        self.unit_data_queue.cancel_join_thread()

        self.compound_data_queue = []
        self.sharedmemory_list = []
        self.compound_data_queue_dict = {}
        self.sharedmemory_dict = {}


        self.compound_data_queue_num = 8

        self.lock = multiprocessing.Lock()
        self.a = np.zeros((dim, 128, 256), dtype=np.float32)
        # if self.rm_equator:
        #     self.a = np.zeros((dim, 720, 1440), dtype=np.float32)
        # else:
        #     self.a = np.zeros((dim, 721, 1440), dtype=np.float32)


        for _ in range(self.compound_data_queue_num):
            self.compound_data_queue.append(multiprocessing.Queue())
            shm = shared_memory.SharedMemory(create=True, size=self.a.nbytes)
            shm.unlink()
            self.sharedmemory_list.append(shm)

        self.arr = multiprocessing.Array('i', range(self.compound_data_queue_num))



        self._workers = []

        for _ in range(60):
            w = multiprocessing.Process(
                target=self.load_data_process)
            w.daemon = True
            # NB: Process.start() actually take some time as it needs to
            #     start a process and pass the arguments over via a pipe.
            #     Therefore, we only add a worker to self._workers list after
            #     it started, so that we do not call .join() if program dies
            #     before it starts, and __del__ tries to join but will get:
            #     AssertionError: can only join a started process.
            w.start()
            self._workers.append(w)
        w = multiprocessing.Process(target=self.data_compound_process)
        w.daemon = True
        w.start()
        self._workers.append(w)


    def init_file_list(self, years):
        time_sequence = pd.date_range(years[0],years[1],freq=str(self.file_stride)+'H') #pd.date_range(start='2019-1-09',periods=24,freq='H')
        self.file_list= [os.path.join(str(time_stamp.year), str(time_stamp.to_datetime64()).split('.')[0]).replace('T', '/')
                      for time_stamp in time_sequence]
        self.single_file_list= [os.path.join('single/'+str(time_stamp.year), str(time_stamp.to_datetime64()).split('.')[0]).replace('T', '/')
                      for time_stamp in time_sequence]


    def _get_meanstd(self):
        with open('/mnt/petrelfs/zhaoxiangyu1/code/weather_prompt_new/dataset/mean_std.json',mode='r') as f:
            multi_level_mean_std = json.load(f)
        with open('/mnt/petrelfs/zhaoxiangyu1/code/weather_prompt_new/dataset/mean_std_single.json',mode='r') as f:
            single_level_mean_std = json.load(f)
        self.mean_std = {}
        multi_level_mean_std['mean'].update(single_level_mean_std['mean'])
        multi_level_mean_std['std'].update(single_level_mean_std['std'])
        self.mean_std['mean'] = multi_level_mean_std['mean']
        self.mean_std['std'] = multi_level_mean_std['std']
        for vname in self.single_level_vnames:
            self.mean_std['mean'][vname] = np.array(self.mean_std['mean'][vname])[::-1][:,np.newaxis,np.newaxis]
            self.mean_std['std'][vname] = np.array(self.mean_std['std'][vname])[::-1][:,np.newaxis,np.newaxis]
        for vname in self.multi_level_vnames:
            self.mean_std['mean'][vname] = np.array(self.mean_std['mean'][vname])[::-1][:,np.newaxis,np.newaxis]
            self.mean_std['std'][vname] = np.array(self.mean_std['std'][vname])[::-1][:,np.newaxis,np.newaxis]

    def get_constants_data(self, constants_types):
        nc_data = xr.open_mfdataset("/mnt/petrelfs/zhaoxiangyu1/code/weather_prompt_new/dataset/constants.nc", 
                                      combine="by_coords", parallel=True)
        array_lst = []
        for vname in constants_types:
            D = nc_data.data_vars[vname].data
            # D = cv2.resize(D, (256, 128), interpolation=cv2.INTER_LINEAR)
            D = standardization(D)
            array_lst.append(D[np.newaxis, :, :])
        data = np.concatenate(array_lst, axis=0)
        array = data
        del array_lst
        return array


    def data_compound_process(self):
        recorder_dict = {}
        while True:
            job_pid, idx, vname, height = self.unit_data_queue.get()
            if job_pid not in self.compound_data_queue_dict:
                try:
                    self.lock.acquire()
                    for i in range(self.compound_data_queue_num):
                        if job_pid == self.arr[i]:
                            self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                            break
                    if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                        print("error", job_pid, self.arr)
                except Exception as err:
                    raise err
                finally:
                    self.lock.release()
            if (job_pid, idx) in recorder_dict:
                recorder_dict[(job_pid, idx)][(vname, height)] = 1
            else:
                recorder_dict[(job_pid, idx)] = {(vname, height): 1}
            if len(recorder_dict[(job_pid, idx)]) == self.data_element_num:
                del recorder_dict[(job_pid, idx)]
                self.compound_data_queue_dict[job_pid].put((idx))


    def get_data(self, idxes):
        job_pid = os.getpid()
        if job_pid not in self.compound_data_queue_dict:
            try:
                self.lock.acquire()
                for i in range(self.compound_data_queue_num):
                    if i == self.arr[i]:
                        self.arr[i] = job_pid
                        self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                        self.sharedmemory_dict[job_pid] = self.sharedmemory_list[i]
                        break
                if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                    print("error", job_pid, self.arr)

                
            except Exception as err:
                raise err
            finally:
                self.lock.release()

        try:
            idx = self.compound_data_queue_dict[job_pid].get(False)
            raise ValueError
        except queue.Empty:
            pass
        except Exception as err:
            raise err
        
        b = np.ndarray(self.a.shape, dtype=self.a.dtype, buffer=self.sharedmemory_dict[job_pid].buf)
        return_data = {}
        for idx in idxes:
            for vname in self.single_level_vnames:
                self.index_queue.put((job_pid, idx, vname, 0))
            for vname in self.multi_level_vnames:
                for height in self.height_level_list:
                    self.index_queue.put((job_pid, idx, vname, height))
            idx = self.compound_data_queue_dict[job_pid].get()
            if self.constants_data is not None:
                return_data[idx] = np.concatenate((self.constants_data, b), axis=0)
            else:
                return_data[idx] = copy.deepcopy(b)
            
        return return_data

    def load_data_process(self):
        while True:
            job_pid, idx, vname, height = self.index_queue.get()
            if job_pid not in self.compound_data_queue_dict:
                try:
                    self.lock.acquire()
                    for i in range(self.compound_data_queue_num):
                        if job_pid == self.arr[i]:
                            self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                            self.sharedmemory_dict[job_pid] = self.sharedmemory_list[i]
                            break
                    if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                        print("error", job_pid, self.arr)
                except Exception as err:
                    raise err
                finally:
                    self.lock.release()
            
            if vname in self.single_level_vnames:
                file = self.single_file_list[idx]
                url = f"{self.data_dir}/{file}-{vname}.npy"

            elif vname in self.multi_level_vnames:
                file = self.file_list[idx]
                url = f"{self.data_dir}/{file}-{vname}-{height}.0.npy"

            b = np.ndarray(self.a.shape, dtype=self.a.dtype, buffer=self.sharedmemory_dict[job_pid].buf)
            with io.BytesIO(self.client.get(url)) as f:
                unit_data = np.load(f)
            unit_data = self.normalization(vname, height, unit_data)

            # unit_data = unit_data[np.newaxis, :, :]
            # if self.rm_equator:
            #     b[self.index_dict1[(vname, height)], :360] = unit_data[:360]
            #     b[self.index_dict1[(vname, height)], 360:] = unit_data[361:]
            # else:
            b[self.index_dict1[(vname, height)], :] = unit_data[:]
            self.unit_data_queue.put((job_pid, idx, vname, height))

    def normalization(self, vname, height, data):
        if vname in self.single_level_vnames:
            index = 0
        else:
            index = height_level.index(height)
        data -=  np.array(self.mean_std['mean'][vname][index], dtype=np.float32)
        data /= np.array(self.mean_std['std'][vname][index],dtype=np.float32)
        return data


    def __len__(self):
        data_len = len(self.file_list) - (self.length - 1) * self.sample_stride
        if self.split == 'valid':
            data_len -= 1
        return data_len

    def get_meanstd(self):
        return_data_mean = []
        return_data_std = []
        
        for vname in self.single_level_vnames:
            return_data_mean.append(self.mean_std['mean'][vname])
            return_data_std.append(self.mean_std['std'][vname])
        for vname in self.multi_level_vnames:
            return_data_mean.append(self.mean_std['mean'][vname][self.height_level_indexes])
            return_data_std.append(self.mean_std['std'][vname][self.height_level_indexes])

        return torch.from_numpy(np.concatenate(return_data_mean, axis=0)[:, 0, 0]), torch.from_numpy(np.concatenate(return_data_std, axis=0)[:, 0, 0])


    def __getitem__(self, index):
        lead_time = [1,4,12,20,28]
        index = min(index, len(self.file_list) - (self.length-1) * self.sample_stride - 1)
        array_dict = self.get_data([index + i * self.sample_stride for i in range(self.length)])
        array_seq = [array_dict[index + i * self.sample_stride] for i in range(self.length)]
        # del array_dict

        padded_img = np.zeros((48, 256, 256), dtype=array_seq[0].dtype)
        padded_img[:, 64:192, :] = array_seq[0]
        if self.lead_time:
            random_time = self.lead_time
        else:
            random_time = random.choice(lead_time)
        label_index = index + random_time
        label_index = min(label_index, len(self.file_list) - (self.length-1) * self.sample_stride - 1)
        label_array_dict = self.get_data([label_index + i * self.sample_stride for i in range(self.length)])
        label_array_seq = [label_array_dict[label_index + i * self.sample_stride] for i in range(self.length)]
        del label_array_dict

        padded_img_label = np.zeros((1, 256, 256), dtype=array_seq[0].dtype)
        padded_img_label[:, 64:192, :] = label_array_seq[0][self.out_var,:,:]

        random_index = random.randint(0, len(self.file_list)-10)
        random_index = min(random_index, len(self.file_list) - (self.length-1) * self.sample_stride - 1)
        context_array_dict = self.get_data([random_index + i * self.sample_stride for i in range(self.length)])
        context_array_seq = [context_array_dict[random_index + i * self.sample_stride] for i in range(self.length)]
        del context_array_dict

        context_padded_img = np.zeros((48, 256, 256), dtype=array_seq[0].dtype)
        context_padded_img[:, 64:192, :] = context_array_seq[0]

        context_label_index = random_index + random_time
        context_label_index = min(context_label_index, len(self.file_list) - (self.length-1) * self.sample_stride - 1)
        context_label_array_dict = self.get_data([context_label_index + i * self.sample_stride for i in range(self.length)])
        context_label_array_seq = [context_label_array_dict[context_label_index + i * self.sample_stride] for i in range(self.length)]
        del context_label_array_dict

        context_padded_img_label = np.zeros((1, 256, 256), dtype=array_seq[0].dtype)
        context_padded_img_label[:, 64:192, :] = context_label_array_seq[0][self.out_var,:,:]

        deg_type = 'era5'
        input_query_img1 = torch.from_numpy(context_padded_img).float()
        target_img1 = torch.from_numpy(context_padded_img_label).float()
        input_query_img2 = torch.from_numpy(padded_img).float()
        target_img2 = torch.from_numpy(padded_img_label).float()

        if self.split == 'train':
            batch = [input_query_img1, target_img1, input_query_img2, target_img2, random_time, self.out_var]
            return batch, deg_type
        else:
            batch = {'input_query_img1': context_padded_img, 'target_img1': context_padded_img_label,
                    'input_query_img2': input_query_img2, 'target_img2': target_img2, 'lead_time': random_time, 'out_var': self.out_var}
            
            return batch, deg_type

class era5_processed(Dataset):
    def __init__(self, data_dir='cluster2:s3://zwl2/era5_processed', split='train', out_var=8, lead_time=None, continue_time=False, **kwargs) -> None:
        super().__init__()
        # self.rm_equator = kwargs.get("rm_equator", True)
        Years_dict = kwargs.get('years', Years)

        self.split = split
        self.out_var = out_var
        self.lead_time = lead_time
        self.data_dir = data_dir
        self.contine_time = continue_time
        self.client = Client(conf_path="~/petreloss.conf")
        # self.client = Client(conf_path="~/petreloss.conf")
        years = Years_dict[split]
        self.init_file_list(years)
    
    def init_file_list(self, years):
        time_sequence = pd.date_range(years[0],years[1],freq=str(6)+'H') #pd.date_range(start='2019-1-09',periods=24,freq='H')
        self.file_list= [os.path.join(str(time_stamp.year), str(time_stamp.to_datetime64()).split('.')[0]).replace('T', '/')
                      for time_stamp in time_sequence]
        self.single_file_list= [os.path.join('single/'+str(time_stamp.year), str(time_stamp.to_datetime64()).split('.')[0]).replace('T', '/')
                      for time_stamp in time_sequence]
        
    def __len__(self):
        data_len = len(self.file_list)
        if self.split == 'valid':
            data_len -= 1
        return data_len
    
    def get_data(self, idxes):

        url = f"{self.data_dir}/{idxes}.npy"
        with io.BytesIO(self.client.get(url)) as f:
            unit_data = np.load(f)
        return unit_data
    
    def __getitem__(self, index):
        lead_time = [1,4,12,20,28]

        index = min(index, len(self.file_list) - 1)

        input_query_img2 = self.get_data(index)
        if self.lead_time:
            random_time = self.lead_time
        else:
            probabilities = [0.3, 0.3, 0.2, 0.1, 0.1]
            # 使用numpy的random.choice函数，按照概率进行抽样
            random_time = np.random.choice(lead_time, p=probabilities)

        label_index = index + random_time
        label_index = min(label_index, len(self.file_list) - 1)
        
        target_img2 = self.get_data(label_index)
        target_img2 = target_img2[self.out_var,:,:]

        if self.contine_time:
            random_index = index-random_time
            random_index = max(random_index, 0)
        else:
            random_index = random.randint(0, len(self.file_list)-10)
            random_index = min(random_index, len(self.file_list) - 1)
        input_query_img1 = self.get_data(random_index)

        context_label_index = random_index + random_time
        context_label_index = min(context_label_index, len(self.file_list) - 1)
        target_img1 = self.get_data(context_label_index)
        target_img1 = target_img1[self.out_var,:,:]

        input_query_img1 = torch.from_numpy(input_query_img1).float()
        target_img1 = torch.from_numpy(target_img1).float().unsqueeze(0)
        input_query_img2 = torch.from_numpy(input_query_img2).float()
        target_img2 = torch.from_numpy(target_img2).float().unsqueeze(0)       

        deg_type = 'era5'

        if self.split == 'train':
            batch = [input_query_img1, target_img1, input_query_img2, target_img2, random_time, self.out_var]
            return batch, deg_type
        else:
            batch = {'input_query_img1': input_query_img1, 'target_img1': target_img1,
                    'input_query_img2': input_query_img2, 'target_img2': target_img2, 'lead_time': random_time, 'out_var': self.out_var}
            
            return batch, deg_type
    




# if __name__ == "__main__":
#     data_set = era5_npy(split='test')

#     for i in range(20):
#         data_set.__getitem__(i)

#     print("complete")


    # for i in range(2220, 2240):
    #     array = data_set._load_array(i)
    # print(array[:, 0, :])
    # print(array[:, -1, :])
    # print(array.shape)
