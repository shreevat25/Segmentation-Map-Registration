import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import UNet, SpatialTransformer
from get_data import SegDataset
from losses import dice_loss, smoothing_loss, composite_loss
from tqdm import tqdm
import numpy as np
import argparse
import os
import wandb
from torch.cuda.amp import GradScaler
torch.cuda.empty_cache()

#logging
wandb.init(project='seg-deformation')
def train(model, stn, dataloader, scaler,scheduler,optimizer, device, epoch, max_epochs=50, dice_weight=1.0, smooth_weight=0.1):
    model.train()
    total_loss = 0
    # dice_loss_total = 0
    # smooth_loss_total = 0
    
    for moving, fixed in tqdm(dataloader, desc=f'Training for epoch: {epoch+1}/{max_epochs}', leave=False):
        moving = moving.to(device)
        fixed = fixed.to(device)
        optimizer.zero_grad()
        input_ = torch.cat([moving, fixed], dim=1)  # Shape: (B, 8, 64, 64, 64), would keep batch size as 1 for now, otherwise might crash
        deformation_field = model(input_)
        
        warped = stn(moving, deformation_field)
        loss = composite_loss(warped, fixed, deformation_field)


        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm=1.0, 
            norm_type=2.0
        )

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
   
        # optimizer.zero_grad()
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        # optimizer.step()

     
        total_loss += loss.item()
        # dice_loss_total += dice_loss_val.item()
        # smooth_loss_total += smooth_loss_val.item()

    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_txt', type=str, default='/local/scratch/v_karthik_mohan/train_npy5.txt', help='Path to the training file listing subject paths')
    #each line contains path to one hot encoded subject
    parser.add_argument('--template_path', type=str, default='/local/scratch/v_karthik_mohan/data/OASIS_OAS1_0406_MR1/seg5_onehot.npy', help='Path to the template segmentation map')
    #for now template is a segmentation map from the dataset itself, ensure it is excluded from training.
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training') 
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--save_model_path', type=str, default='/local/scratch/v_karthik_mohan/code-base/weights2/', help='Path to save the trained model')
    args = parser.parse_args()

    device = 'cuda:3' #change accordingly

  
    print("Loading dataset")
    train_dataset = SegDataset(args.train_txt, args.template_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print("Dataset loaded.")
    scaler = GradScaler()
    # U-Net to predict deformations, STN to warp the deformations on top of the template
    model = UNet(in_channels=10, out_channels=3).to(device)
    stn = SpatialTransformer(size=(128,128,128), device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=args.epochs*len(train_loader),
    eta_min=1e-5
    )


    for epoch in range(args.epochs):
        avg_loss= train(
            model, stn, train_loader, scaler, scheduler,optimizer, device, epoch,args.epochs
        )
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.4f}")
        
        wandb.log({
            'epoch': epoch + 1,
            'loss': avg_loss,
           
        })
        #save the model every 10 epochs
        if (epoch + 1) % 50 == 0:
            model_path = f'{args.save_model_path}_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    
    torch.save(model.state_dict(), args.save_model_path)
    print(f"Model saved to {args.save_model_path}")

if __name__ == "__main__":
    main()
