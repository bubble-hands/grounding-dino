import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


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
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, features):
        B, C, H, W = features[0].shape
        flat_features = [f.flatten(2).transpose(1, 2) for f in features]

        gate_weights = []
        for f in flat_features:
            w = self.gate(f)
            gate_weights.append(w)

        gate_weights = torch.stack(gate_weights, dim=2)
        gate_weights = gate_weights.mean(dim=-1)
        gate_weights = F.softmax(gate_weights, dim=-1)

        attended_features = torch.stack(flat_features, dim=2)
        fused = (attended_features * gate_weights.unsqueeze(-1)).sum(dim=2)

        return fused.transpose(1, 2).view(B, C, H, W)


class SwinBackbone(nn.Module):
    def __init__(self, cfg, in_chans=3):
        super().__init__()
        self.cfg = cfg
        self.output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS

        model_name = 'swin_tiny_patch4_window7_224'
        use_pretrained = cfg.MODEL.get('PRETRAINED', False)
        
        # 预训练模型通常是 3 通道 RGB，如果输入通道不同，需要特殊处理
        self.adapter_in_chans = in_chans
        if in_chans != 3 and use_pretrained:
            print(f"[SwinBackbone] Input channels={in_chans} != 3, using conv adapter + pretrained Swin")
            self.input_adapter = nn.Conv2d(in_chans, 3, kernel_size=1)
            in_chans_for_swin = 3
        else:
            self.input_adapter = None
            in_chans_for_swin = in_chans

        if HAS_TIMM:
            try:
                self.backbone = timm.create_model(
                    model_name,
                    pretrained=use_pretrained,
                    features_only=True,
                    out_indices=(0, 1, 2, 3),
                    img_size=512,
                    in_chans=in_chans_for_swin,
                )
                feature_info = self.backbone.feature_info
                self.feature_channels = [feature_info.channels()[i] for i in range(4)]
                init_type = "pretrained" if use_pretrained else "random init"
                print(f"[SwinBackbone] Created {model_name} ({init_type}, in_chans={in_chans}, img_size=512), channels: {self.feature_channels}")
            except Exception as e:
                print(f"[SwinBackbone] Cannot create Swin ({e}), using conv backbone")
                self.backbone = None
                self.feature_channels = [96, 192, 384, 768]
        else:
            print("[SwinBackbone] timm not available, using conv backbone")
            self.backbone = None
            self.feature_channels = [96, 192, 384, 768]

        self.projections = nn.ModuleList()
        for in_ch, out_ch in zip(self.feature_channels, self.output_channels):
            self.projections.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch),
                nn.GELU()
            ))

    def forward(self, x):
        if self.backbone is not None:
            # 如果有 input_adapter，先将输入转换为 3 通道
            if self.input_adapter is not None:
                x = self.input_adapter(x)
            features = self.backbone(x)
            features = [f.permute(0, 3, 1, 2).contiguous() for f in features]
            projected = [proj(f) for proj, f in zip(self.projections, features)]
            return projected
        else:
            return None


class SimpleBackbone(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        self.cfg = cfg
        self.output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS

        self.layers = nn.ModuleList()
        prev_dim = in_channels
        for i, dim in enumerate(self.output_channels):
            self.layers.append(nn.Sequential(
                nn.Conv2d(prev_dim, dim, kernel_size=3, padding=1, stride=2 if i > 0 else 1),
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim),
                nn.GELU()
            ))
            prev_dim = dim

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features


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
        self.use_swin = cfg.MODEL.get('USE_SWIN', True)

        self.adapters = nn.ModuleDict()
        for modality in self.modalities:
            in_chans = self.input_channels[modality]
            self.adapters[modality] = ModalityAdapter(
                in_channels=in_chans,
                out_channels=self.adapter_dim
            )

        if self.use_swin:
            self.swin_backbone = SwinBackbone(cfg, in_chans=self.adapter_dim)
            self.use_swin_impl = self.swin_backbone.backbone is not None
        else:
            self.use_swin_impl = False

        self.fusion_modules = nn.ModuleList()
        for dim in self.output_channels:
            self.fusion_modules.append(MultiModalFusion(dim, len(self.modalities)))

    def forward(self, inputs, text_features=None):
        if self.use_swin_impl:
            return self._forward_swin(inputs)
        else:
            return self._forward_simple(inputs)

    def _forward_swin(self, inputs):
        modality_features = {}

        for modality in self.modalities:
            if modality in inputs and inputs[modality] is not None:
                feat = self.adapters[modality](inputs[modality])
                features = self.swin_backbone(feat)
                modality_features[modality] = features

        fusion_out = []
        for level in range(len(self.output_channels)):
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

    def _forward_simple(self, inputs):
        modality_features = {}

        for modality in self.modalities:
            if modality in inputs and inputs[modality] is not None:
                feat = self.adapters[modality](inputs[modality])
                backbone = SimpleBackbone(self.cfg, self.adapter_dim)
                features = backbone(feat)
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