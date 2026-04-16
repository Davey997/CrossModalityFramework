import torch
import torch.nn as nn
import importlib

from model.backbone import DualModalityBackbone
from model.my_fusion_module import FusionModule, My_fusion_module
from model.my_seg_head import HybridAttentionDecodeHead
from model.yolox_head import YOLOXHead 

class Model_proposal(nn.Module):
    def __init__(self,
                 backbone: dict,
                 head: dict,
                 fusion: dict,
                 name: str = "model_proposal",
                 freeze: bool = False,
                 **kwargs):
        super().__init__()

        self.name = name  # 'model_proposal' is already the default arg

        #DualModalityBackbone remains the same
        self.backbone = DualModalityBackbone(
            rgb_backbone=backbone.get('name', 'resnet50'),
            event_backbone=backbone.get('name', 'resnet50'),
            outputs=["preflatten_feat", "flatten_feat"],
            img_size=backbone.get('input_size', 512),
            embed_dim=backbone.get('embed_dim', 256),
            output_indices=backbone.get('output_indices', [3, 4])
        )

        #Read channel info per scale
        img_feature_info = self.backbone.rgb_backbone.feature_info
        evt_feature_info = self.backbone.event_backbone.feature_info
        out_indices = backbone['output_indices']
        
        in_ch_image = [i['num_chs'] for i in img_feature_info if i['index'] in out_indices]
        in_ch_event = [i['num_chs'] for i in evt_feature_info if i['index'] in out_indices]
        
        strides = [i['reduction'] for i in img_feature_info if i['index'] in out_indices]
        out_ch_fusion = fusion.get('out_channels', 256)

        #One FusionModule per scale
        self.fusion_modules = nn.ModuleList([
            FusionModule(
                in_channels_image=ic_img,
                in_channels_event=ic_evt,
                out_channels=out_ch_fusion,
                num_heads=fusion.get('num_heads', 8),
                dropout=fusion.get('dropout', 0.1)
            )
            for ic_img, ic_evt in zip(in_ch_image, in_ch_event)
        ])

        #Decoder head
        self.decoder = YOLOXHead(
            num_classes=head['num_classes'],
            in_channels=[out_ch_fusion] * len(out_indices),
            strides=strides,
            losses_weights=head.get('losses_weights', [5.0, 1.0, 1.0, 1.0])
        )

        #Optional freezing — same pattern as resnet50_yolox.py
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def get_name(self):
        return self.name

    def forward(self, rgb, events, targets=None):
        img_dict, evt_dict = self.backbone(rgb, events)
        img_feats = img_dict["preflatten_feat"]
        evt_feats = evt_dict["preflatten_feat"]

        fused = [
            fusion_mod(img_f, evt_f)
            for fusion_mod, img_f, evt_f in zip(self.fusion_modules, img_feats, evt_feats)
        ]

        head_predictions, total_loss, losses_dict = self.decoder(fused, labels=targets)

        return {
            'backbone_features': {
                'preflatten_feat': img_feats,
                'flatten_feat': img_dict.get('flatten_feat', None)
            },
            'head_outputs': head_predictions,
            'total_loss': total_loss if total_loss is not None else torch.tensor(0.0, requires_grad=True),
            'losses': losses_dict if losses_dict is not None else {}
        }