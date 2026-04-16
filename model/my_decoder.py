import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCustomHead(nn.Module):
    def __init__(self, **kwargs):
        super(MyCustomHead, self).__init__(**kwargs)
        self.conv1 = nn.Conv2d(self.in_channels, 2048, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(2048, self.num_classes, kernel_size=1)

        self.attention = nn.Sequential(
          nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1),
          nn.Sigmoid()
        )

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        attention_map = self.attention(x)
        x = x * attention_map
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.conv2(x)

        return out


class DualAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(DualAttentionBlock, self).__init__()
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
    def __init__(self, in_channels=2048, channels=256, fusion_channels=256, **kwargs):
        super().__init__(in_channels=in_channels, channels=channels, **kwargs)

        self.attn_img = DualAttentionBlock(self.in_channels)
        self.attn_evt = DualAttentionBlock(self.in_channels)
        self.attn_fused = DualAttentionBlock(self.in_channels)

        self.fusion = nn.Sequential(
            nn.Conv2d(self.in_channels * 3, fusion_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.cls_seg = nn.Conv2d(fusion_channels, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        assert isinstance(inputs, (list, tuple)) and len(inputs) == 3
        x_img = self.attn_img(inputs[0])
        x_evt = self.attn_evt(inputs[1])
        x_fused = self.attn_fused(inputs[2])

        x = torch.cat([x_img, x_evt, x_fused], dim=1)

        x = self.fusion(x)
        out = self.cls_seg(x)

        return out
