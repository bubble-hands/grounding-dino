import torch
import torch.nn as nn
from .swin_transformer import SwinTransformer
from .modality_adapter import ModalityAdapter, MultiModalFeatureFusion


class MultiModalVisualBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.modalities = config.MODEL.MULTI_MODAL.MODALITIES
        self.input_channels = config.MODEL.MULTI_MODAL.INPUT_CHANNELS
        self.adapter_dim = config.MODEL.MULTI_MODAL.ADAPTER_DIM
        self.out_channels = config.MODEL.BACKBONE.OUT_CHANNELS

        self.modality_adapters = nn.ModuleDict()
        for modality in self.modalities:
            in_chans = self.input_channels[modality]
            self.modality_adapters[modality] = ModalityAdapter(
                in_channels=in_chans,
                embed_dim=self.adapter_dim
            )

        self.shared_backbone = SwinTransformer(
            img_size=config.INPUT.SIZE_TRAIN[0],
            patch_size=4,
            in_chans=self.adapter_dim,
            embed_dim=self.adapter_dim,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.1,
            drop_path_rate=0.1
        )

        self.fusion_modules = nn.ModuleList()
        for dim in self.out_channels:
            self.fusion_modules.append(
                MultiModalFeatureFusion(embed_dim=dim, num_modalities=len(self.modalities))
            )

        self.text_guided_gate = TextGuidedModalityGate(
            text_dim=config.MODEL.TEXT_ENCODER.DIM,
            num_modalities=len(self.modalities)
        )

    def forward(self, inputs, text_features=None):
        modality_features = {}

        for modality in self.modalities:
            if modality in inputs and inputs[modality] is not None:
                adapter_out = self.modality_adapters[modality](inputs[modality])
                _, features = self.shared_backbone(adapter_out)
                modality_features[modality] = features

        fused_features = []
        for level, fusion_module in enumerate(self.fusion_modules):
            level_features = []
            for modality in self.modalities:
                if modality in modality_features:
                    feat, H, W = modality_features[modality][level]
                    feat = feat.permute(0, 2, 1).view(-1, self.out_channels[level], H, W)
                    level_features.append(feat)

            if len(level_features) > 0:
                if text_features is not None and self.config.MODEL.MULTI_MODAL.USE_TEXT_GUIDANCE:
                    gate_weights = self.text_guided_gate(text_features)
                    weighted_features = [
                        f * gate_weights[:, i].view(-1, 1, 1, 1)
                        for i, f in enumerate(level_features)
                    ]
                    fused = fusion_module(weighted_features)
                else:
                    fused = fusion_module(level_features)
                fused_features.append(fused)

        return fused_features


class TextGuidedModalityGate(nn.Module):
    def __init__(self, text_dim=768, num_modalities=3):
        super().__init__()
        self.proj = nn.Linear(text_dim, num_modalities)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, text_features):
        cls_token = text_features[:, 0, :]
        weights = self.proj(cls_token)
        return self.softmax(weights)