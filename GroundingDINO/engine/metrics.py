"""
训练监控指标工具：mAP 计算 + MetricsLogger 日志记录。

- box_iou_cxcywh:  计算 cxcywh 归一化框的 IoU
- compute_map:     单类 mAP@IoU 计算 (VOC all-points 插值)
- MetricsLogger:   按 batch / epoch 记录 loss、mAP、lr 到 CSV + JSONL
"""
import os
import csv
import json
import torch
import numpy as np
from datetime import datetime


def box_cxcywh_to_xyxy(boxes):
    """cxcywh -> xyxy, 支持张量和 ndarray"""
    if isinstance(boxes, torch.Tensor):
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    cx, cy, w, h = np.split(np.asarray(boxes, dtype=np.float64), 4, axis=-1)
    return np.concatenate([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def box_iou_xyxy(boxes1, boxes2):
    """boxes1: [N,4] xyxy, boxes2: [M,4] xyxy -> IoU [N, M] (ndarray)"""
    b1 = np.asarray(boxes1, dtype=np.float64)
    b2 = np.asarray(boxes2, dtype=np.float64)
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    lt = np.maximum(b1[:, None, :2], b2[None, :, :2])
    rb = np.minimum(b1[:, None, 2:], b2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, 1e-8, None)


def compute_map(predictions, ground_truths, iou_threshold=0.5):
    """单类 mAP@IoU 计算。

    Args:
        predictions:   list[dict], 每项 {'boxes': [N,4] cxcywh, 'scores': [N]}
        ground_truths: list[dict], 每项 {'boxes': [M,4] cxcywh}
        iou_threshold: IoU 阈值

    Returns:
        ap (float): average precision at given IoU threshold
    """
    n_gt = sum(len(gt['boxes']) for gt in ground_truths)
    if n_gt == 0:
        return 0.0

    # 展开所有预测: (score, img_idx, box_xyxy)
    all_scores = []
    all_img = []
    all_boxes = []
    for img_idx, pred in enumerate(predictions):
        boxes = pred['boxes']
        scores = pred['scores']
        if len(scores) == 0:
            continue
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        xyxy = box_cxcywh_to_xyxy(boxes)
        all_scores.append(scores)
        all_img.append(np.full(len(scores), img_idx))
        all_boxes.append(xyxy)

    if len(all_scores) == 0:
        return 0.0

    all_scores = np.concatenate(all_scores)
    all_img = np.concatenate(all_img)
    all_boxes = np.concatenate(all_boxes)

    # 按分数降序排列
    order = np.argsort(-all_scores)
    all_scores = all_scores[order]
    all_img = all_img[order]
    all_boxes = all_boxes[order]

    # GT 转 xyxy, 记录每张图的 GT 框
    gt_xyxy_list = []
    for gt in ground_truths:
        gb = gt['boxes']
        if isinstance(gb, torch.Tensor):
            gb = gb.cpu().numpy()
        gt_xyxy_list.append(box_cxcywh_to_xyxy(gb) if len(gb) > 0 else np.zeros((0, 4)))

    matched = [np.zeros(len(g), dtype=bool) for g in gt_xyxy_list]
    tp = np.zeros(len(all_scores))
    fp = np.zeros(len(all_scores))

    for i in range(len(all_scores)):
        img_idx = all_img[i]
        gt_boxes = gt_xyxy_list[img_idx]
        if len(gt_boxes) == 0:
            fp[i] = 1
            continue
        ious = box_iou_xyxy(all_boxes[i:i + 1], gt_boxes)[0]  # [M]
        best_iou = ious.max()
        best_j = ious.argmax()
        if best_iou >= iou_threshold and not matched[img_idx][best_j]:
            tp[i] = 1
            matched[img_idx][best_j] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.clip(tp_cum + fp_cum, 1e-8, None)

    # VOC all-points interpolation
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ap = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))
    return ap


class MetricsLogger:
    """训练指标日志记录器，写 CSV (便于解析) + JSONL (便于阅读)。"""

    BATCH_COLUMNS = ['timestamp', 'epoch', 'batch_idx', 'lr',
                     'loss', 'loss_ce', 'loss_bbox', 'loss_giou']
    EPOCH_COLUMNS = ['timestamp', 'epoch', 'train_loss', 'val_loss',
                     'mAP50', 'lr', 'elapsed_s']

    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.batch_csv = os.path.join(log_dir, 'metrics_batch.csv')
        self.epoch_csv = os.path.join(log_dir, 'metrics_epoch.csv')
        self.jsonl_path = os.path.join(log_dir, 'metrics.jsonl')

        with open(self.batch_csv, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(self.BATCH_COLUMNS)
        with open(self.epoch_csv, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(self.EPOCH_COLUMNS)

    def log_batch(self, epoch, batch_idx, lr, loss, loss_dict=None):
        """记录单 batch 训练指标。loss_dict 可选, 含 loss_ce/loss_bbox/loss_giou。"""
        row = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'epoch': epoch,
            'batch_idx': batch_idx,
            'lr': f'{lr:.2e}',
            'loss': f'{loss:.6f}',
            'loss_ce': f'{loss_dict["loss_ce"].item():.6f}' if loss_dict and 'loss_ce' in loss_dict else '',
            'loss_bbox': f'{loss_dict["loss_bbox"].item():.6f}' if loss_dict and 'loss_bbox' in loss_dict else '',
            'loss_giou': f'{loss_dict["loss_giou"].item():.6f}' if loss_dict and 'loss_giou' in loss_dict else '',
        }
        with open(self.batch_csv, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([row[c] for c in self.BATCH_COLUMNS])

    def log_epoch(self, epoch, train_loss, val_loss, mAP50, lr, elapsed_s):
        """记录单 epoch 汇总指标。"""
        row = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'epoch': epoch,
            'train_loss': f'{train_loss:.6f}',
            'val_loss': f'{val_loss:.6f}' if val_loss is not None else '',
            'mAP50': f'{mAP50:.6f}' if mAP50 is not None else '',
            'lr': f'{lr:.2e}',
            'elapsed_s': f'{elapsed_s:.1f}',
        }
        with open(self.epoch_csv, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([row[c] for c in self.EPOCH_COLUMNS])

        record = {
            'timestamp': row['timestamp'],
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'mAP50': mAP50,
            'lr': lr,
            'elapsed_s': round(elapsed_s, 1),
        }
        with open(self.jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
