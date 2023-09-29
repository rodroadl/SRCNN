'''
dataset.py
source: https://github.com/yjn870/SRCNN-pytorch/blob/master/datasets.py
modification: name of the file

GunGyeom James Kim
September 28th, 2023
CS 7180: Advnaced Perception

code for custom dataset
'''

import h5py
import numpy as np
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, h5_file):
        '''
        constructor

        Parameters:
            h5_file - path for data
        '''
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        '''
        Return an image and label for given index

        Parameters:
            idx(int): index

        Return:
            image
            label
        '''
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx]/ 255., 0)

    def __len__(self):
        '''
        Return the length of the dataset

        Return:
            length(int)
        '''
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

class EvalDataset(Dataset):
    def __init__(self, h5_file):
        '''
        constructor

        Parameters:
            h5_file - path for data
        '''
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file
    
    def __getitem__(self, idx):
        '''
        Return an image and label for given index

        Parameters:
            idx(int): index

        Return:
            image
            label
        '''
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)
        
    def __len__(self):
        '''
        Return the length of the dataset

        Return:
            length(int)
        '''
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])