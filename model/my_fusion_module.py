import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from model.backbone import UnimodalBackbone, DualModalityBackbone
from model.yolox_head import YOLOXHead

import random

   
class FusionModule(nn.Module):
    def __init__(self, in_channels_image, in_channels_event, out_channels,
                 num_heads=8, dropout=0.1, **kwargs):
        super().__init__()

        # Project both modalities to common dimension
        self.img_proj = nn.Linear(in_channels_image, out_channels)
        self.evt_proj = nn.Linear(in_channels_event, out_channels)

        # Cross-attention: events (queries) attend to image (keys/values)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Post-attention norm + feedforward
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 4, out_channels),
            nn.Dropout(dropout)
        )

    def forward(self, image_features, event_features):
        B, C_img, H, W = image_features.shape

        #Flatten spatial dims into tokens
        img_tokens = image_features.flatten(2).permute(0, 2, 1)  # [B, HW, C_img]
        evt_tokens = event_features.flatten(2).permute(0, 2, 1)  # [B, HW, C_evt]

        #Project to common dim
        img_tokens = self.img_proj(img_tokens)  # [B, HW, D]
        evt_tokens = self.evt_proj(evt_tokens)  # [B, HW, D]

        # Cross-attention: evt queries, img keys/values
        attn_out, _ = self.cross_attn(
            query=evt_tokens,
            key=img_tokens,
            value=img_tokens
        )

        #Residual + norm (pre-norm style)
        evt_tokens = self.norm1(evt_tokens + attn_out)

        #FFN + residual
        evt_tokens = self.norm2(evt_tokens + self.ffn(evt_tokens))

        #Reshape back to spatial feature map
        fused = evt_tokens.permute(0, 2, 1).reshape(B, -1, H, W)  # [B, D, H, W]
        return fused


class My_fusion_module(nn.Module):
    def __init__(self,
                 image_backbone: dict,
                 event_backbone: dict,
                 name: str = "fusion_seg_model",
                 head: dict = {'name': 'seg_head', 'num_classes': 8},
                 freeze: bool = False,
                 **kwargs):
        super().__init__()

        #Init both backbones (heads)
        self.backbone = DualModalityBackbone(
            rgb_backbone=image_backbone.get('name', 'resnet50'),
            event_backbone=event_backbone.get('name', 'resnet50'),
            outputs=["preflatten_feat"],
            img_size=image_backbone.get('input_size', 512)
        )

        #Read channel info 
        img_feature_info = self.backbone.rgb_backbone.feature_info
        evt_feature_info = self.backbone.event_backbone.feature_info
        out_indices = image_backbone['output_indices']

        in_ch_image = [i['num_chs'] for i in img_feature_info if i['index'] in out_indices]
        in_ch_event = [i['num_chs'] for i in evt_feature_info if i['index'] in out_indices]
        out_ch_fusion = head.get('fusion_channels', 256)

        # One FusionModule per scale ("fusion module")
        self.fusion_modules = nn.ModuleList([
            FusionModule(in_channels_image=ic_img,
                        in_channels_event=ic_evt,
                        out_channels=out_ch_fusion)
            for ic_img, ic_evt in zip(in_ch_image, in_ch_event)
        ])

    def forward(self, image, events):
        img_dict, evt_dict = self.backbone(image, events)
        img_feats = img_dict["preflatten_feat"]
        evt_feats = evt_dict["preflatten_feat"]
        
        return [
            fusion(img_f, evt_f)
            for fusion, img_f, evt_f in zip(self.fusion_modules, img_feats, evt_feats)
        ]
        # → [tensor(B, 256, 16, 16), tensor(B, 256, 8, 8)]