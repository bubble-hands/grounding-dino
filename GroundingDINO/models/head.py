import torch
import torch.nn as nn
import torch.nn.functional as F


class GroundingHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.class_head = nn.Sequential(
            nn.Linear(cfg.MODEL.HIDDEN_DIM, cfg.MODEL.HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(cfg.MODEL.HIDDEN_DIM, cfg.MODEL.HIDDEN_DIM)
        )

        self.box_head = nn.Sequential(
            nn.Linear(cfg.MODEL.HIDDEN_DIM, cfg.MODEL.HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(cfg.MODEL.HIDDEN_DIM, cfg.MODEL.HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(cfg.MODEL.HIDDEN_DIM, 4)
        )

        self.text_proj = nn.Linear(cfg.MODEL.TEXT_ENCODER.DIM, cfg.MODEL.HIDDEN_DIM)

    def forward(self, queries, text_features):
        class_feat = self.class_head(queries)
        box_pred = self.box_head(queries)

        text_proj = self.text_proj(text_features)

        sim = torch.matmul(class_feat, text_proj.transpose(1, 2))

        box_pred = torch.sigmoid(box_pred)

        return sim, box_pred