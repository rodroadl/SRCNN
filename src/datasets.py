import logging 
import os
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, data_dir:Path):
        self.data_dir = data_dir
        self.ext = splitext(os.listdir(self.data_dir)[0])[1]
        self.ids = [splitext(file)[0] for file in os.listdir(self.data_dir) if isfile(join(self.data_dir, file)) and not file.startswith('.')]
        if not self.ids:
            return RuntimeError(f'No input file found in {self.data_dir}, make sure you put your images there')
        
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __getitem__(self, idx):
        img_path = join(self.data_dir, self.ids[idx], self.ext)
        img = read_image(img_path)
        return img

    def __len__(self):
        return len(self.ids)