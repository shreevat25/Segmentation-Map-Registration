import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )

        def upsample_block(in_channels, out_channels):
            return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

        # Encoder
        self.enc1 = conv_block(in_channels, 32)  # 8 channels: template + fixed given together as input 4 channels each
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc4 = conv_block(128, 256)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        
        self.bottleneck = conv_block(256, 512)

        # Decoder
        self.up4 = upsample_block(512, 256)
        self.dec4 = conv_block(512, 256)
        self.up3 = upsample_block(256, 128)
        self.dec3 = conv_block(256, 128)
        self.up2 = upsample_block(128, 64)
        self.dec2 = conv_block(128, 64)
        self.up1 = upsample_block(64, 32)
        self.dec1 = conv_block(64, 32)

        
        self.out_conv = nn.Conv3d(32, 3, kernel_size=1)  # 3 channels for deformation field (x, y, z)

    def forward(self, x):
        # Encoder part
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        #Decoder part
        up4 = self.up4(b)
        d4 = self.dec4(torch.cat((up4, e4), dim=1))
        up3 = self.up3(d4)
        d3 = self.dec3(torch.cat((up3, e3), dim=1))
        up2 = self.up2(d3)
        d2 = self.dec2(torch.cat((up2, e2), dim=1))
        up1 = self.up1(d2)
        d1 = self.dec1(torch.cat((up1, e1), dim=1))

     
        deformation_field = self.out_conv(d1)
        return deformation_field

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTransformer(nn.Module):
    """
    A true dense STN for 3D one-hot maps, with an identity grid buffer
    and nearest‐neighbor sampling for crisp labels.
    """
    def __init__(self, size, device='cpu'):
        """
        Args:
            size: tuple of ints (D, H, W)
            device: tensor device
        """
        super().__init__()
        D, H, W = size
 
        lin_z = torch.linspace(-1, 1, D, device=device)
        lin_y = torch.linspace(-1, 1, H, device=device)
        lin_x = torch.linspace(-1, 1, W, device=device)
        zz, yy, xx = torch.meshgrid(lin_z, lin_y, lin_x, indexing='ij')

        id_grid = torch.stack((xx, yy, zz), dim=-1)
        self.register_buffer('id_grid', id_grid.unsqueeze(0))

    def forward(self, moving, flow):
        """
        Args:
            moving: (B, C, D, H, W) one‐hot template
            flow:   (B, 3, D, H, W) displacement in normalized coords
        Returns:
            warped: (B, C, D, H, W) one‐hot warped template
        """
        B, C, D, H, W = moving.shape
      
        flow = flow.permute(0, 2, 3, 4, 1)
   
        grid = self.id_grid.expand(B, -1, -1, -1, -1)
    
        warped_grid = grid + flow
      
        warped = F.grid_sample(
            moving, warped_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )
        return warped
