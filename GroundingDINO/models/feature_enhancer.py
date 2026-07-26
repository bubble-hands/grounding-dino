import torch
import torch.nn as nn
import torch.nn.functional as F

from .text_encoder import TextEncoder


class FeatureEnhancer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.text_encoder = TextEncoder(
            model_name=cfg.MODEL.TEXT_ENCODER.NAME,
            hidden_dim=cfg.MODEL.TEXT_ENCODER.DIM
        )

        self.image_to_text_attns = nn.ModuleList()
        self.text_to_image_attns = nn.ModuleList()
        self.image_projs = nn.ModuleList()
        
        for in_dim in cfg.MODEL.BACKBONE.OUT_CHANNELS:
            self.image_projs.append(nn.Linear(in_dim, cfg.MODEL.HIDDEN_DIM))
            self.image_to_text_attns.append(nn.MultiheadAttention(
                embed_dim=cfg.MODEL.HIDDEN_DIM,
                num_heads=min(cfg.MODEL.NECK.NUM_HEADS, cfg.MODEL.HIDDEN_DIM // 32),
                batch_first=True,
                dropout=0.1
            ))
            self.text_to_image_attns.append(nn.MultiheadAttention(
                embed_dim=cfg.MODEL.HIDDEN_DIM,
                num_heads=min(cfg.MODEL.NECK.NUM_HEADS, cfg.MODEL.HIDDEN_DIM // 32),
                batch_first=True,
                dropout=0.1
            ))

        self.text_self_attn = nn.MultiheadAttention(
            embed_dim=cfg.MODEL.TEXT_ENCODER.DIM,
            num_heads=min(cfg.MODEL.NECK.NUM_HEADS, cfg.MODEL.TEXT_ENCODER.DIM // 32),
            batch_first=True,
            dropout=0.1
        )

        self.text_norm = nn.LayerNorm(cfg.MODEL.TEXT_ENCODER.DIM)
        self.image_norm = nn.LayerNorm(cfg.MODEL.HIDDEN_DIM)

        self.text_proj = nn.Linear(cfg.MODEL.TEXT_ENCODER.DIM, cfg.MODEL.HIDDEN_DIM)

    def forward(self, image_features, text_input_ids, text_attention_mask):
        text_features = self.text_encoder(text_input_ids, text_attention_mask)
        text_features = self.text_norm(text_features)
        text_features, _ = self.text_self_attn(text_features, text_features, text_features)

        text_proj = self.text_proj(text_features)

        enhanced_features = []
        for i, feat in enumerate(image_features):
            B, C, H, W = feat.shape
            flat = feat.flatten(2).transpose(1, 2)
            
            flat = self.image_projs[i](flat)

            it_out, _ = self.image_to_text_attns[i](flat, text_proj, text_proj)
            ti_out, _ = self.text_to_image_attns[i](flat, text_proj, text_proj)

            it_out = it_out.transpose(1, 2).view(B, self.cfg.MODEL.HIDDEN_DIM, H, W)
            ti_out = ti_out.transpose(1, 2).view(B, self.cfg.MODEL.HIDDEN_DIM, H, W)

            enhanced = flat.transpose(1, 2).contiguous().view(B, self.cfg.MODEL.HIDDEN_DIM, H, W) + it_out + ti_out
            enhanced = self.image_norm(enhanced.flatten(2).transpose(1, 2)).transpose(1, 2).contiguous().view(B, self.cfg.MODEL.HIDDEN_DIM, H, W)
            enhanced_features.append(enhanced)

        return enhanced_features, text_features