import torch
from torch.utils.data import Dataset
import numpy as np
import os

class SegDataset(Dataset):
    def __init__(self, data_list_file: str, template_path: str):
   
        # Read subject paths from txt file where each line is a subejct path
        with open(data_list_file, 'r') as file:
            self.subject_paths = file.read().splitlines()

        # Load the moving template segmentation map
        self.moving_template = np.load(template_path)
        self.moving_template = torch.tensor(self.moving_template, dtype=torch.float32)

    def __len__(self):
        return len(self.subject_paths)

    def __getitem__(self, idx):
    
        fixed_path = self.subject_paths[idx]
        fixed_map = np.load(fixed_path)

        # Convert to tensors
        fixed_map = torch.tensor(fixed_map, dtype=torch.float32)

        return self.moving_template, fixed_map


