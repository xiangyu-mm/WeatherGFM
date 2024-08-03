import os
import glob
import h5py
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

class SingleFrameDataset(Dataset):
    def __init__(self, img_list):
        self.img_list = img_list
        self.frame_count = 49  
        print(f'num ims: {len(img_list)}')

    def __len__(self):
        return len(self.img_list) * self.frame_count

    def __getitem__(self, index):
        file_index = index // self.frame_count
        frame_index = index % self.frame_count

        file_name = self.img_list[file_index]
        eventFile = h5py.File(file_name, 'r')

        ir069 = eventFile['ir069'][:, :, frame_index]
        ir107 = eventFile['ir107'][:, :, frame_index]
        vis = eventFile['vis'][:, :, frame_index]
        vil = eventFile['vil'][:, :, frame_index]

        ir069 = np.expand_dims(ir069, axis=0)
        ir107 = np.expand_dims(ir107, axis=0)
        vis = np.expand_dims(vis, axis=0)
        vil = np.expand_dims(vil, axis=0)

        ir069 = torch.from_numpy(ir069.astype(np.float32))
        ir107 = torch.from_numpy(ir107.astype(np.float32))
        vis = torch.from_numpy(vis.astype(np.float32))
        vil = torch.from_numpy(vil.astype(np.float32))

        return ir069, ir107, vis, vil
    
    
###############################################
# GET DATASET USING PYTORCH LIGHTNING
###############################################

class SEVIRDataModule(pl.LightningDataModule):
    """define __init__, setup, and loaders for 3sets of data!"""
    def __init__(self, 
                 # data_params, training_params, 
                data_dir,
                batch_size,
                ):
        super().__init__() 
        # self.params = params

        ## Let's define the fileList here!
        trainFilesTotal = glob.glob(data_dir+'train/*') 

        self.testFiles = glob.glob(data_dir+'test/*')

        self.trainFiles = trainFilesTotal[:8000]
        self.valFiles = trainFilesTotal[8000:]
        self.batch_size = batch_size
        
    def setup(self, stage=None):
        if stage == 'train' or stage is None:
            self.train_dataset = SingleFrameDataset(img_list=self.trainFiles)
            self.val_dataset = SingleFrameDataset(img_list=self.valFiles)

        if stage == 'test':
            self.test_dataset = SingleFrameDataset(img_list=self.testFiles)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        # No shuffling!
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        # Batch size is 1 for testing! No shuffling!
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=4)