# get_data.py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os
#for now resize one hot encoded segmentation maps down to 64^3 otherwise will face memory issues
class SegDataset(Dataset):
    def __init__(self, data_list_file: str, template_path: str, target_size=(64, 64, 64)):
        # Read subject paths from txt file where each line is a subject path
        with open(data_list_file, 'r') as file:
            self.subject_paths = file.read().splitlines()

        # Load the moving template segmentation map
        self.moving_template = np.load(template_path)
        self.moving_template = torch.tensor(self.moving_template, dtype=torch.float32)

  
        self.moving_template = F.interpolate(
            self.moving_template.unsqueeze(0),  
            size=target_size,
            mode='trilinear',  
            align_corners=False
        ).squeeze(0)  

        self.target_size = target_size

    def __len__(self):
        return len(self.subject_paths)

    def __getitem__(self, idx):
        fixed_path = self.subject_paths[idx]
        fixed_map = np.load(fixed_path)

        
        fixed_map = torch.tensor(fixed_map, dtype=torch.float32)
        fixed_map = F.interpolate(
            fixed_map.unsqueeze(0),
            size=self.target_size,
            mode='trilinear',  
            align_corners=False
        ).squeeze(0)  

        return self.moving_template, fixed_map
