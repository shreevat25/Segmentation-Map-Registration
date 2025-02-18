import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os

class SegDataset(Dataset):
    def __init__(self, data_list_file: str, template_path: str, target_size=(64, 64, 64)):
        # Read subject paths from txt file where each line is a subject path
        with open(data_list_file, 'r') as file:
            self.subject_paths = file.read().splitlines()

        # Load the moving template segmentation map
        self.moving_template = np.load(template_path)
        self.moving_template = torch.tensor(self.moving_template, dtype=torch.float32)

        # Resize the template to the target size
        self.moving_template = F.interpolate(
            self.moving_template.unsqueeze(0),  # Add batch dimension
            size=target_size,
            mode='nearest'  # Use 'nearest' for one-hot encoded segmentations
        ).squeeze(0)  # Remove batch dimension

        self.target_size = target_size

    def __len__(self):
        return len(self.subject_paths)

    def __getitem__(self, idx):
        fixed_path = self.subject_paths[idx]
        fixed_map = np.load(fixed_path)

        # Convert to tensor and resize
        fixed_map = torch.tensor(fixed_map, dtype=torch.float32)
        fixed_map = F.interpolate(
            fixed_map.unsqueeze(0),  # Add batch dimension
            size=self.target_size,
            mode='nearest'  # Use 'nearest' for one-hot encoded segmentations
        ).squeeze(0)  # Remove batch dimension

        return self.moving_template, fixed_map
