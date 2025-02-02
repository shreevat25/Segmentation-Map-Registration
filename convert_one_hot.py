import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

# Paths
input_txt = "/local/scratch/v_karthik_mohan/template.txt"

labels = [0, 1, 2, 3]  
# Labels in thisc case are Background, Cortex, Subcortical GM, White Matter, CSF

# Function to convert to one-hot encoding
def to_one_hot(segmentation, labels):
    one_hot = np.zeros((len(labels), *segmentation.shape), dtype=np.uint8)  # Shape: (4, H, W, D)
    for i, label in enumerate(labels):
        one_hot[i] = (segmentation == label).astype(np.uint8)
    return one_hot

# Process each segmentation map
with open(input_txt, 'r') as file:
    paths = file.read().splitlines()

for path in tqdm(paths, desc="Processing segmentation maps"):
    # Load segmentation map
    img = nib.load(path)
    seg_map = img.get_fdata().astype(np.int16) 
    one_hot_map = to_one_hot(seg_map, labels)

    # Saving in same subject directory
    output_path = path.replace("seg4.nii.gz", "seg4_onehot.npy")

    np.save(output_path, one_hot_map)

