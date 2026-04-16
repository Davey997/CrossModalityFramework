import torch
import torch.nn as nn
import torch.nn.functional as F


class DualAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        sa = self.spatial_attn(x)
        ca = self.channel_attn(x)
        return x * sa * ca


class HybridAttentionDecodeHead(nn.Module):
    def __init__(self, in_channels_raw=2048, in_channels_fused=256, 
                 fusion_channels=256, num_classes=8, img_size=512, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size

        # Raw streams use backbone channels (2048)
        self.attn_img = DualAttentionBlock(in_channels_raw)
        self.attn_evt = DualAttentionBlock(in_channels_raw)
        
        # Fused stream uses fusion module output channels (256)
        self.attn_fused = DualAttentionBlock(in_channels_fused)

        # Concatenation: 2048 + 2048 + 256
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels_raw * 2 + in_channels_fused, fusion_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.cls_seg = nn.Conv2d(fusion_channels, num_classes, kernel_size=1)

    def forward(self, img_feats, evt_feats, fused_feats, targets=None):
        x_img   = self.attn_img(img_feats[-1])      # [B, 2048, H, W]
        x_evt   = self.attn_evt(evt_feats[-1])      # [B, 2048, H, W]
        x_fused = self.attn_fused(fused_feats[-1])  # [B, 256,  H, W]

        x = torch.cat([x_img, x_evt, x_fused], dim=1)  # [B, 2048+2048+256, H, W]
        x = self.fusion(x)
        out = self.cls_seg(x)
        out = F.interpolate(out, size=(self.img_size, self.img_size),
                            mode='bilinear', align_corners=False)

        if targets is not None:
            return out, self.compute_loss(out, targets)
        return out, None

    def compute_loss(self, pred, targets):
        return F.cross_entropy(pred, targets.long())