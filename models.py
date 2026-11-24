import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import log_change

# Check for mamba-ssm
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    log_change("mamba_ssm successfully imported for SS2DBlocks.", "ARCHITECTURE")
except ImportError:
    MAMBA_AVAILABLE = False
    log_change("WARNING: mamba_ssm is not installed. Using a fallback Conv layer.", "WARNING")

class FallbackSS2D(nn.Module):
     # Since native Mamba failed to compile, we inject a heavy ConvNeXt Block 
     # to ensure deep global context and high parameter utilization.
     def __init__(self, d_model):
         super().__init__()
         self.dwconv = nn.Conv2d(d_model, d_model, kernel_size=7, padding=3, groups=d_model) # Depthwise
         self.norm = nn.BatchNorm2d(d_model)
         self.pwconv1 = nn.Conv2d(d_model, 4 * d_model, kernel_size=1) # Pointwise upscale
         self.act = nn.GELU()
         self.pwconv2 = nn.Conv2d(4 * d_model, d_model, kernel_size=1) # Pointwise downscale

     def forward(self, x):
         input_x = x
         x = self.dwconv(x)
         x = self.norm(x)
         x = self.pwconv1(x)
         x = self.act(x)
         x = self.pwconv2(x)
         return x + input_x

class SS2DBlock(nn.Module): 
    # 2D Selective Scan (Mamba) for linear global context
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=d_model, 
                d_state=d_state, 
                d_conv=d_conv, 
                expand=expand
            )
        else:
            self.mamba = FallbackSS2D(d_model)

    def forward(self, x):
        if not MAMBA_AVAILABLE:
            return x + self.mamba(x)

        B, C, H, W = x.shape
        # Flatten spatially: [B, C, H, W] -> [B, H*W, C] ensures alignment with Mamba sequence expectations
        x_flat = x.flatten(2).transpose(1, 2)
        
        mamba_out = self.mamba(x_flat) 
        
        # Reshape back to [B, C, H, W]
        out = mamba_out.transpose(1, 2).view(B, C, H, W)
        return x + out 

class HybridEncoder(nn.Module): 
    # CNN early stages -> SS2D bottleneck stages
    def __init__(self, in_channels=14, base_dim=32):
        super().__init__()
        # Input is [B, 14, 64, 64]
        
        # 1. Local Texture CNN Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
            # Downsample to 32x32
            nn.Conv2d(base_dim, base_dim*2, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(base_dim*2),
            nn.ReLU(inplace=True)
        )
        
        # 2. Mid-level CNN (Downsample to 16x16)
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_dim*2, base_dim*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_dim*4),
            nn.ReLU(inplace=True)
        )
        
        # 3. Mamba Global Context Bottleneck (Operates on 16x16 spatial grid)
        self.mamba_bottleneck = nn.Sequential(
            SS2DBlock(base_dim*4),
            SS2DBlock(base_dim*4)
        )

    def forward(self, x):
        # [B, 64, 32, 32]
        feat_high_res = self.stem(x)
        # [B, 128, 16, 16]
        feat_low_res = self.stage2(feat_high_res)
        # [B, 128, 16, 16]
        feat_global = self.mamba_bottleneck(feat_low_res)
        return feat_high_res, feat_global


class EvidentialDecoder(nn.Module): 
    # Upsampling decoder ending in Softplus (Evidence >= 0)
    def __init__(self, base_dim=32, num_classes=2):
        super().__init__()
        
        # Upsample global features to 32x32 to match high_res
        self.up1 = nn.ConvTranspose2d(base_dim*4, base_dim*2, kernel_size=4, stride=2, padding=1) 
        
        # Fusion layer (concatenate high-res CNN + Upsampled Mamba)
        self.fusion = nn.Sequential(
            nn.Conv2d(base_dim*4, base_dim*2, kernel_size=3, padding=1), # [B, 128, 32, 32] -> [B, 64, 32, 32]
            nn.BatchNorm2d(base_dim*2),
            nn.ReLU(inplace=True)
        )
        
        # Final Upsample back to 64x64
        self.up2 = nn.ConvTranspose2d(base_dim*2, base_dim, kernel_size=4, stride=2, padding=1) 
        
        # Output Head
        self.head = nn.Conv2d(base_dim, num_classes, kernel_size=1)
        self.softplus = nn.Softplus()

    def forward(self, feat_high_res, feat_global, return_features=False):
        x_up = self.up1(feat_global)
        
        # Ensure spatial alignment mathematically before concatenation
        if x_up.shape[2:] != feat_high_res.shape[2:]:
             x_up = F.interpolate(x_up, size=feat_high_res.shape[2:], mode="bilinear", align_corners=False)

        fused = torch.cat([x_up, feat_high_res], dim=1)
        fused = self.fusion(fused)
        
        x_out = self.up2(fused)
        logits = self.head(x_out)
        
        evidence = self.softplus(logits)
        
        if return_features:
            return evidence, x_out
        
        return evidence

class FADENet(nn.Module):
    def __init__(self, in_channels=14, num_classes=2, base_dim=128):
        super().__init__()
        self.encoder = HybridEncoder(in_channels=in_channels, base_dim=base_dim)
        self.decoder = EvidentialDecoder(base_dim=base_dim, num_classes=num_classes)
        log_change(f"Initialized FADE-Net (base_dim={base_dim}). Input Channels: {in_channels}", "ARCHITECTURE")

    def forward(self, x, return_features=False):
        feat_high_res, feat_global = self.encoder(x)
        return self.decoder(feat_high_res, feat_global, return_features=return_features)
