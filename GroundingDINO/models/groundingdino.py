import torch
import torch.nn as nn

from .backbone import MultiModalBackbone
from .feature_enhancer import FeatureEnhancer
from .decoder import GroundingDINOTransformerDecoder
from .head import GroundingHead
from .losses import SetCriterion, HungarianMatcher


class GroundingDINO(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.backbone = MultiModalBackbone(cfg)
        self.feature_enhancer = FeatureEnhancer(cfg)
        self.decoder = GroundingDINOTransformerDecoder(cfg)
        
        self.class_head = nn.Sequential(
            nn.Linear(cfg.MODEL.HIDDEN_DIM, cfg.MODEL.HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(cfg.MODEL.HIDDEN_DIM, 1)
        )

        self.box_head = nn.Sequential(
            nn.Linear(cfg.MODEL.HIDDEN_DIM, cfg.MODEL.HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(cfg.MODEL.HIDDEN_DIM, cfg.MODEL.HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(cfg.MODEL.HIDDEN_DIM, 4)
        )

        self.text_proj = nn.Linear(cfg.MODEL.TEXT_ENCODER.DIM, cfg.MODEL.HIDDEN_DIM)

        matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
        weight_dict = {"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2}
        losses = ["labels", "boxes"]
        self.criterion = SetCriterion(num_classes=1, matcher=matcher, weight_dict=weight_dict,
                                      eos_coef=0.1, losses=losses)

    def forward(self, inputs):
        image_inputs = {
            'rgb': inputs.get('rgb', None),
            'ir': inputs.get('ir', None),
            'depth': inputs.get('depth', None)
        }

        text_input_ids = inputs['text_input_ids']
        text_attention_mask = inputs['text_attention_mask']

        image_features = self.backbone(image_inputs)

        enhanced_features, text_features = self.feature_enhancer(
            image_features, text_input_ids, text_attention_mask
        )

        queries = self.decoder(enhanced_features, text_features)

        text_proj = self.text_proj(text_features)

        B, N, C = queries.shape
        T = text_proj.shape[1]

        class_feat = self.class_head(queries)

        sim = torch.matmul(queries, text_proj.transpose(1, 2))
        sim = sim.mean(dim=-1, keepdim=True)
        pred_logits = torch.cat([class_feat, sim], dim=-1)

        box_pred = self.box_head(queries)
        box_pred = torch.sigmoid(box_pred)

        outputs = {
            'pred_logits': pred_logits,
            'pred_boxes': box_pred
        }

        if self.training and 'targets' in inputs:
            loss = self.criterion(outputs, inputs['targets'])
            outputs['loss'] = sum(loss.values())

        return outputs

    def inference(self, inputs):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs)
        return {
            'pred_logits': outputs['pred_logits'].cpu().numpy(),
            'pred_boxes': outputs['pred_boxes'].cpu().numpy()
        }