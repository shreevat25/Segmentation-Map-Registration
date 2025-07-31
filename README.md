# Unsupervised Segmentation Map Registration

This repository implements an unsupervised deep learning framework for registering a surface-derived segmentation template to a target segmentation volume. The method learns a 3D deformation field using volumetric similarity and smoothness constraints. It supports multiple surface-aware loss functions and works on medical datasets such as OASIS.

---

## Dataset

We use the [Neurite-OASIS](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md) brain MRI dataset. The `.npz` formatted data can be downloaded as described in the linked repo. Segmentation maps are converted to one-hot encoded `.npy` files for training.

---

##  Setup

```bash
git clone https://github.com/karthiknm/Segmentation-Map-Registration.git
cd Segmentation-Map-Registration
pip install -r requirements.txt  # or install dependencies manually
```

---

## Training

To train the model, run:

```bash
python train.py --train_txt path/to/train_npy_list.txt \
                --template_path path/to/template_seg_onehot.npy \
                --epochs 200 \
                --save_model_path ./weights/
```

---

## File Overview

- `train.py`: Main training loop using U-Net and spatial transformer.
- `model.py`: Defines 3D U-Net and `SpatialTransformer`.
- `losses.py`: Includes Dice loss, Chamfer, Jacobian, and other geometric losses.
- `compoundlossfunction.py`: Combines multiple loss functions with user-defined weights.
- `convert_one_hot.py`: Converts `.nii.gz` segmentation files to one-hot `.npy` format.
- `get_data.py`: Custom PyTorch `Dataset` for loading one-hot segmentation maps.
- `testing_loss.py`: Unit tests for validating loss functions.

---

## Loss Functions Used

- Dice Loss  
- Chamfer Distance Loss  
- Hausdorff Distance Loss  
- Bending Energy Loss  
- Jacobian Determinant Loss  
- Label Overlap Loss  
- Deformation Direction Variation

These are configured and combined in `compoundlossfunction.py`.

---

## Notes

- A fixed one-hot encoded surface template is warped to match each fixed segmentation map.
- `SpatialTransformer` applies the deformation field using bilinear or nearest-neighbor sampling.
- You may need to edit file paths, device assignments, and wandb settings in `train.py`.

---

## Logging

Training progress is logged using [Weights & Biases (wandb)](https://wandb.ai/). To disable it, comment out the `wandb.init()` line in `train.py`.

---

##  Example Command

```bash
python train.py --train_txt ./data/train_npy5.txt \
                --template_path ./data/OASIS_OAS1_0406_MR1/seg5_onehot.npy \
                --epochs 200 \
                --save_model_path ./weights/model.pth
```
