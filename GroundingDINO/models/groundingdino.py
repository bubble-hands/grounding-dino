import torch
import torch.nn as nn
import torchvision.ops as ops

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

        # P0-2: 平衡分类与定位，避免 class_head 塌缩
        # cost_class=5 让 matcher 在匹配时充分考虑分类，cost_bbox/cost_giou 保持较高但不压制分类
        matcher = HungarianMatcher(cost_class=5, cost_bbox=8, cost_giou=6)
        weight_dict = {"loss_ce": 5, "loss_bbox": 8, "loss_giou": 6}
        losses = ["labels", "boxes"]
        self.criterion = SetCriterion(num_classes=1, matcher=matcher, weight_dict=weight_dict,
                                      eos_coef=0.1, losses=losses)

        # P0-4: 初始化 box_head 最后一层 bias，使初始预测框接近 GT 统计尺寸 (w≈0.1, h≈0.2)
        # sigmoid(bias) ≈ 目标值 → bias = logit(目标值) = ln(target / (1 - target))
        # cx≈0.5 → bias_cx = 0, cy≈0.5 → bias_cy = 0
        # w≈0.1 → bias_w = ln(0.1/0.9) ≈ -2.197
        # h≈0.2 → bias_h = ln(0.2/0.8) ≈ -1.386
        import math
        with torch.no_grad():
            last_layer = self.box_head[-1]
            last_layer.bias.copy_(torch.tensor([
                0.0,           # cx → sigmoid(0) = 0.5
                0.0,           # cy → sigmoid(0) = 0.5
                math.log(0.1 / 0.9),  # w  → sigmoid(-2.197) ≈ 0.1
                math.log(0.2 / 0.8),  # h  → sigmoid(-1.386) ≈ 0.2
            ]))
            print(f"[GroundingDINO] box_head bias initialized to small-target prior: "
                  f"cx=0.5, cy=0.5, w=0.1, h=0.2")

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

        if 'targets' in inputs:
            loss = self.criterion(outputs, inputs['targets'])
            # 仅对 weight_dict 中的 loss 项加权求和，排除 class_error 等纯指标
            weight_dict = self.criterion.weight_dict
            outputs['loss'] = sum(
                loss[k] * weight_dict[k]
                for k in loss.keys() if k in weight_dict
            )
            outputs['loss_dict'] = loss

        if not self.training:
            outputs = self.postprocess(outputs)

        return outputs

    def postprocess(self, outputs, score_threshold=0.3, nms_threshold=0.5, topk=10, single_box=False):
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']

        scores = torch.sigmoid(pred_logits).squeeze(-1)
        max_scores, _ = scores.max(dim=-1)

        batch_results = []
        for b in range(scores.shape[0]):
            box = pred_boxes[b]
            score = max_scores[b]

            if single_box:
                best_idx = torch.argmax(score)
                best_box = box[best_idx:best_idx+1]
                best_score = score[best_idx:best_idx+1]
                batch_results.append({
                    'boxes': best_box,
                    'scores': best_score
                })
            else:
                keep = score > score_threshold
                filtered_boxes = box[keep]
                filtered_scores = score[keep]

                if filtered_boxes.shape[0] == 0:
                    topk_idx = torch.topk(score, min(topk, score.shape[0])).indices
                    filtered_boxes = box[topk_idx]
                    filtered_scores = score[topk_idx]

                if filtered_boxes.shape[0] > topk:
                    topk_idx = torch.topk(filtered_scores, topk).indices
                    filtered_boxes = filtered_boxes[topk_idx]
                    filtered_scores = filtered_scores[topk_idx]

                if filtered_boxes.shape[0] > 0:
                    nms_idx = ops.nms(filtered_boxes, filtered_scores, nms_threshold)
                    filtered_boxes = filtered_boxes[nms_idx]
                    filtered_scores = filtered_scores[nms_idx]

                batch_results.append({
                    'boxes': filtered_boxes,
                    'scores': filtered_scores
                })

        outputs['results'] = batch_results
        return outputs

    def inference(self, inputs, score_threshold=0.3, nms_threshold=0.5, topk=10, single_box=False):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs)
        
        if 'results' in outputs:
            results = outputs['results']
            return [{
                'boxes': r['boxes'].cpu().numpy(),
                'scores': r['scores'].cpu().numpy()
            } for r in results]
        
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        scores = torch.sigmoid(pred_logits).squeeze(-1)
        max_scores, _ = scores.max(dim=-1)
        
        if single_box:
            batch_results = []
            for b in range(scores.shape[0]):
                best_idx = torch.argmax(max_scores[b])
                batch_results.append({
                    'boxes': pred_boxes[b][best_idx:best_idx+1].cpu().numpy(),
                    'scores': max_scores[b][best_idx:best_idx+1].cpu().numpy()
                })
            return batch_results
        
        return {
            'pred_logits': outputs['pred_logits'].cpu().numpy(),
            'pred_boxes': outputs['pred_boxes'].cpu().numpy()
        }