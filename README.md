**Unsupervised Segmentation Map Registration**

This repository implements an unsupervised deep learning framework for registering a surface-derived segmentation template to a target segmentation volume. The model learns a 3D deformation field using volumetric similarity and smoothness constraints, without requiring paired training data. It supports multiple surface-aware loss functions and operates on medical datasets such as OASIS.

---

**Dataset**

We use the [Neurite-OASIS](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md) brain MRI dataset. The `.npz` files are preprocessed into one-hot encoded `.npy` volumes using `convert_one_hot.py`. Each one-hot volume has **5 channels**, corresponding to:  
- Background  
- Cortex  
- Subcortical GM  
- White Matter  
- CSF

---

**Setup**

```bash
git clone https://github.com/karthiknm/Segmentation-Map-Registration.git
cd Segmentation-Map-Registration
pip install -r requirements.txt  # or install dependencies manually
```

---

**Training**

```bash
python train.py --train_txt path/to/train_npy_list.txt \
                --template_path path/to/template_seg_onehot.npy \
                --epochs 200 \
                --save_model_path ./weights/
```

The input to the model is constructed by concatenating the 5-channel template and 5-channel fixed map, resulting in a **10-channel input**. The U-Net is initialized with `in_channels=10`.

---

**File Overview**

- `train.py`: Main training loop using U-Net and `SpatialTransformer`.
- `model.py`: Contains both the 3D U-Net and a `SpatialTransformer` using bilinear sampling.
- `losses.py`: Defines all atomic losses used for training (`dice_loss`, `bending_energy_loss`, `jacobian_det_loss`, etc.).
- `compoundlossfunction.py`: Combines multiple losses using a `loss_weights` dictionary — useful for experimentation, but **not used in the main training loop**.
- `get_data.py`: Loads `.npy` one-hot maps and resizes them to `(128, 128, 128)` using **nearest-neighbor interpolation**.
- `convert_one_hot.py`: Converts `.nii.gz` segmentations into 5-channel `.npy` format.
- `testing_loss.py`: Unit test to verify individual loss functions return valid outputs.

---

**Loss Functions Used**

The main training script (`train.py`) uses a predefined `composite_loss` function from `losses.py`, which includes:
- Dice Loss
- Cross Entropy Loss
- Bending Energy Loss
- Jacobian Determinant Loss

In addition, `compoundlossfunction.py` supports experimental loss combinations using:
- Chamfer Distance
- Hausdorff Distance
- Surface Distance
- Label Overlap Loss
- Deformation Direction Variation

---

**Notes**

- One fixed template is warped to match each subject's segmentation.
- The `SpatialTransformer` performs dense warping using bilinear interpolation on tensors in normalized coordinates.
- Label-preserving nearest-neighbor interpolation is used only during dataset resizing (`get_data.py`).
- Be sure to adjust paths and GPU IDs in `train.py` as needed.

---

**Logging**

Training progress is tracked using [Weights & Biases (wandb)](https://wandb.ai/). To disable, comment out or remove the `wandb.init()` call in `train.py`.

---

**Example Command**

```bash
python train.py --train_txt ./data/train_npy5.txt \
                --template_path ./data/OASIS_OAS1_0406_MR1/seg5_onehot.npy \
                --epochs 200 \
                --save_model_path ./weights/model.pth
```

---

