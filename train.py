import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import UNet, SpatialTransformer
from get_data import SegDataset
from losses import dice_loss, smoothing_loss
from tqdm import tqdm
import numpy as np
import argparse
import os
torch.cuda.empty_cache()
# train.py
def train(model, stn, dataloader, optimizer, device, dice_weight=1.0, smooth_weight=0.1):
    model.train()
    total_loss = 0
    dice_loss_total = 0
    smooth_loss_total = 0

    for moving, fixed in tqdm(dataloader, desc='Training Batches', leave=False):  # Fix: Correct variable names
        moving = moving.to(device)
        fixed = fixed.to(device)

        # Concatenate moving and fixed along channels
        input_to_model = torch.cat([moving, fixed], dim=1)  # Shape: (B, 8, 256, 256, 256)
        
        # Forward pass through U-Net
        deformation_field = model(input_to_model)
        
        # Warp the moving image (template)
        warped_template = stn(moving, deformation_field)

        # Compute losses
        dice_loss_val = dice_loss(warped_template, fixed)
        smooth_loss_val = smoothing_loss(deformation_field)
        loss = dice_weight * dice_loss_val + smooth_weight * smooth_loss_val

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        dice_loss_total += dice_loss_val.item()
        smooth_loss_total += smooth_loss_val.item()

    return total_loss / len(dataloader), dice_loss_total / len(dataloader), smooth_loss_total / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_txt', type=str, default = '/local/scratch/v_karthik_mohan/train_npy.txt', help='Path to the training file listing subject paths')
    parser.add_argument('--template_path', type=str, default = '/local/scratch/v_karthik_mohan/data/OASIS_OAS1_0406_MR1/seg4_onehot.npy', help='Path to the template segmentation map')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--save_model_path', type=str, default='./trained_model.pth', help='Path to save the trained model')
    args = parser.parse_args()

    device = 'cuda:3'

    # Load dataset 
    print("Loading dataset")
    train_dataset = SegDataset(args.train_txt, args.template_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print("Loading template. ")
    # Load template: for now using a median file from the dataset itself as a template, can be adjusted.
    template = torch.from_numpy(np.load(args.template_path)).float().unsqueeze(0).to(device)  # Shape (1, 4, 256, 256, 256)

    
    model = UNet(in_channels=8, out_channels=3).to(device)
    stn = SpatialTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in tqdm(range(args.epochs), desc='Training Epochs'):
        avg_loss, avg_dice, avg_smooth = train(
            model, stn, train_loader, optimizer, template, device
        )

        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.4f}, Dice Loss: {avg_dice:.4f}, Smoothing Loss: {avg_smooth:.4f}")

    torch.save(model.state_dict(), args.save_model_path)
    print(f"Model saved to {args.save_model_path}")

if __name__ == "__main__":
    main()
