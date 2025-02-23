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
import wandb
torch.cuda.empty_cache()

#logging
wandb.init(project='seg-deformation')
def train(model, stn, dataloader, optimizer, device, epoch, max_epochs=50, dice_weight=1.0, smooth_weight=0.1):
    model.train()
    total_loss = 0
    dice_loss_total = 0
    smooth_loss_total = 0
    
    for moving, fixed in tqdm(dataloader, desc=f'Training for epoch: {epoch+1}/{max_epochs}', leave=False):
        moving = moving.to(device)
        fixed = fixed.to(device)

        input_ = torch.cat([moving, fixed], dim=1)  # Shape: (B, 8, 64, 64, 64), would keep batch size as 1 for now, otherwise might crash
       
        deformation_field = model(input_)
        
        warped_template = stn(moving, deformation_field)

        dice_loss_val = dice_loss(warped_template, fixed)
        smooth_loss_val = smoothing_loss(deformation_field)
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
    parser.add_argument('--train_txt', type=str, default='/local/scratch/v_karthik_mohan/train_npy.txt', help='Path to the training file listing subject paths')
    #each line contains path to one hot encoded subject
    parser.add_argument('--template_path', type=str, default='/local/scratch/v_karthik_mohan/data/OASIS_OAS1_0406_MR1/seg4_onehot.npy', help='Path to the template segmentation map')
    #for now template is a segmentation map from the dataset itself, ensure it is excluded from training.
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training') 
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--save_model_path', type=str, default='./trained_model.pth', help='Path to save the trained model')
    args = parser.parse_args()

    device = 'cuda:3' #change accordingly

  
    print("Loading dataset")
    train_dataset = SegDataset(args.train_txt, args.template_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print("Dataset loaded.")

    # U-Net to predict deformations, STN to warp the deformations on top of the template
    model = UNet(in_channels=8, out_channels=3).to(device)
    stn = SpatialTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    for epoch in range(args.epochs):
        avg_loss, avg_dice, avg_smooth = train(
            model, stn, train_loader, optimizer, device, epoch,args.epoch
        )
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.4f}, Dice Loss: {avg_dice:.4f}, Smoothing Loss: {avg_smooth:.4f}")
        
        wandb.log({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'dice_loss': avg_dice,
            'smoothing_loss': avg_smooth
        })
        #save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            model_path = f'{args.save_model_path}_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    
    torch.save(model.state_dict(), args.save_model_path)
    print(f"Model saved to {args.save_model_path}")

if __name__ == "__main__":
    main()
