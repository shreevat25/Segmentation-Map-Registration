 import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.encoder = nn.ModuleDict({
            'enc1': self._block(in_channels, 32),
            'enc2': self._block(32, 64),
            'enc3': self._block(64, 128),
            'enc4': self._block(128, 256)
        })
        
        self.bottleneck = self._block(256, 512)
        self.pool = nn.MaxPool3d(2, 2)
        self.decoder = nn.ModuleDict({
            'dec4': self._block(512, 256),
            'dec3': self._block(256, 128),
            'dec2': self._block(128, 64),
            'dec1': self._block(64, 32)
        })
        
        self.upsample = nn.ModuleDict({
            'up4': self._upsample(512, 256),
            'up3': self._upsample(256, 128),
            'up2': self._upsample(128, 64),
            'up1': self._upsample(64, 32)
        })
        
        self.out_conv = nn.Sequential(
            nn.Conv3d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 3, 1, bias = False),
            nn.Tanh()
        )

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.2)
        )
    
    def _upsample(self, in_ch, out_ch):
        return nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)


    def forward(self, x):
        # Encoder
        e1 = self.encoder['enc1'](x)
        p1 = self.pool(e1)
        
        e2 = self.encoder['enc2'](p1)
        p2 = self.pool(e2)
        
        e3 = self.encoder['enc3'](p2)
        p3 = self.pool(e3)
        
        e4 = self.encoder['enc4'](p3)
        p4 = self.pool(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder
        up4 = self.upsample['up4'](b)
        d4 = self.decoder['dec4'](torch.cat([up4, e4], dim=1))
        
        up3 = self.upsample['up3'](d4)
        d3 = self.decoder['dec3'](torch.cat([up3, e3], dim=1))
        
        up2 = self.upsample['up2'](d3)
        d2 = self.decoder['dec2'](torch.cat([up2, e2], dim=1))
        
        up1 = self.upsample['up1'](d2)
        d1 = self.decoder['dec1'](torch.cat([up1, e1], dim=1))

        # Final output
        return self.out_conv(d1)
    
class SpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, moving, deformation):
        # Generate grid dynamically for device compatibility
        B, _, H, W, D = moving.shape
        grid = F.affine_grid(
            torch.eye(3,4, device=moving.device).unsqueeze(0).repeat(B,1,1),
            moving.shape,
            align_corners=False
        )

       
        warped_grid = grid + deformation.permute(0,2,3,4,1) 
        warped = F.grid_sample(
            moving,
            warped_grid,
            mode='bilinear',  # 
            padding_mode='border',
            align_corners=False
        )
        return warped 
