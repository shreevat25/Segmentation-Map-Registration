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

def train(model, stn, dataloader, optimizer, template, device, dice_weight=1.0, smooth_weight=0.1):
    model.train()
    total_loss = 0
    dice_loss_total = 0
    smooth_loss_total = 0

    for fixed_seg_map, fixed_seg_path in tqdm(dataloader, desc='Training Batches', leave=False):
        fixed_seg_map = fixed_seg_map.to(device)
        template = template.to(device)

        # Forward pass through U-Net
        deformation_field = model(template)
        
        # get warped template
        warped_template = stn(template, deformation_field)

        dice_loss_val = dice_loss(warped_template, fixed_seg_map)
        smooth_loss_val = smoothing_loss(deformation_field)
        
        # Composite loss function - for now the weights of the losses are hyperparameters, future they can also be
        #learnt
        loss = dice_weight * dice_loss_val + smooth_weight * smooth_loss_val

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        dice_loss_total += dice_loss_val.item()
        smooth_loss_total += smooth_loss_val.item()

    return total_loss / len(dataloader), dice_loss_total / len(dataloader), smooth_loss_total / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_txt', type=str, required=True, help='Path to the training file listing subject paths', default = '/local/scratch/v_karthik_mohan/train_npy.txt')
    parser.add_argument('--template_path', type=str, required=True, help='Path to the template segmentation map', default = '/local/scratch/v_karthik_mohan/data/OASIS_OAS1_0406_MR1/seg4_onehot.npy')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--save_model_path', type=str, default='./trained_model.pth', help='Path to save the trained model')
    args = parser.parse_args()

    device = 'cuda'

    # Load dataset 
    train_dataset = SegDataset(args.train_txt)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Load template: for now using a median file from the dataset itself as a template, can be adjusted.
    template = torch.from_numpy(np.load(args.template_path)).float().unsqueeze(0).to(device)  # Shape (1, 4, 256, 256, 256)

    
    model = UNet(in_channels=4, out_channels=3).to(device)
    stn = SpatialTransformer(size=(256, 256, 256)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in tqdm(range(args.epochs), desc='Training Epochs'):
        avg_loss, avg_dice, avg_smooth = train(
            model, stn, train_loader, optimizer, template, device
        )

        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.4f}, Dice Loss: {avg_dice:.4f}, Smoothing Loss: {avg_smooth:.4f}")

    torch.save(model.state_dict(), args.save_model_path)
    print(f"Model saved to {args.save_model_path}")
