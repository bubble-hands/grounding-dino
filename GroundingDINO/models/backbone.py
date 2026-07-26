import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformer


class ModalityAdapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class MultiModalFusion(nn.Module):
    def __init__(self, dim, num_modalities):
        super().__init__()
        self.dim = dim
        self.num_modalities = num_modalities
        
        self.gate = nn.Sequential(
            nn.Linear(dim * num_modalities, dim),
            nn.Sigmoid()
        )

    def forward(self, features):
        B, C, H, W = features[0].shape
        flat_features = [f.flatten(2).transpose(1, 2) for f in features]
        
        concat_features = torch.cat(flat_features, dim=2)
        gate_weights = self.gate(concat_features).unsqueeze(2)
        
        attended_features = torch.stack(flat_features, dim=2)
        fused = (attended_features * gate_weights).sum(dim=2)
        
        return fused.transpose(1, 2).view(B, C, H, W)


class MultiModalBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.modalities = cfg.MODEL.MULTI_MODAL.MODALITIES
        self.input_channels = {
            'rgb': cfg.MODEL.MULTI_MODAL.INPUT_CHANNELS_RGB,
            'ir': cfg.MODEL.MULTI_MODAL.INPUT_CHANNELS_IR,
            'depth': cfg.MODEL.MULTI_MODAL.INPUT_CHANNELS_DEPTH
        }
        self.adapter_dim = cfg.MODEL.MULTI_MODAL.ADAPTER_DIM
        self.output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS

        self.adapters = nn.ModuleDict()
        for modality in self.modalities:
            in_chans = self.input_channels[modality]
            self.adapters[modality] = ModalityAdapter(
                in_channels=in_chans,
                out_channels=self.adapter_dim
            )

        self.backbone_layers = nn.ModuleList()
        prev_dim = self.adapter_dim
        for i, dim in enumerate(self.output_channels):
            self.backbone_layers.append(nn.Sequential(
                nn.Conv2d(prev_dim, dim, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim),
                nn.GELU()
            ))
            prev_dim = dim

        self.fusion_modules = nn.ModuleList()
        for dim in self.output_channels:
            self.fusion_modules.append(MultiModalFusion(dim, len(self.modalities)))

    def forward(self, inputs, text_features=None):
        modality_features = {}

        for modality in self.modalities:
            if modality in inputs and inputs[modality] is not None:
                feat = self.adapters[modality](inputs[modality])
                
                features = []
                for i, layer in enumerate(self.backbone_layers):
                    if i == 0:
                        x = layer(feat)
                    else:
                        x = layer(x)
                    features.append(x)
                
                modality_features[modality] = features

        fusion_out = []
        for level in range(4):
            level_features = []
            for modality in self.modalities:
                if modality in modality_features:
                    level_features.append(modality_features[modality][level])
            
            if level_features:
                fused = self.fusion_modules[level](level_features)
                fusion_out.append(fused)
            else:
                fusion_out.append(None)

        return fusion_out