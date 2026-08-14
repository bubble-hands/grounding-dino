"""
单批过拟合测试 - 验证模型是否能在少量样本上正常训练
如果模型不能在1个batch上收敛，说明存在严重的代码bug
"""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groundingdino.config.GroundingDINO_Fused_Train import get_cfg
from groundingdino.datasets.dataset import MultiModalDataset, MultiModalCollator
from groundingdino.models.groundingdino import GroundingDINO


def test_single_batch_overfit():
    cfg = get_cfg()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 创建数据集和数据加载器
    train_dataset = MultiModalDataset(cfg, split='train')
    collator = MultiModalCollator(cfg)
    
    # 只取一个batch
    from torch.utils.data import DataLoader
    loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )
    
    batch = next(iter(loader))
    
    # 移到设备
    inputs = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
        else:
            inputs[k] = v
    
    print(f"\nBatch 结构:")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}, dtype={v.dtype}")
        elif isinstance(v, list):
            print(f"  {k}: list of {len(v)} items")
            if len(v) > 0:
                print(f"    - 第一个元素: {v[0]}")
    
    # 创建模型
    model = GroundingDINO(cfg).to(device)
    model.train()
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # 训练循环
    print(f"\n{'='*60}")
    print(f"开始单批过拟合测试 (100 iterations)")
    print(f"{'='*60}")
    
    losses = []
    loss_ces = []
    loss_bboxes = []
    loss_gious = []
    
    for step in range(100):
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = outputs['loss']
        loss_dict = outputs.get('loss_dict', {})
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
        loss_ces.append(loss_dict.get('loss_ce', torch.tensor(0)).item())
        loss_bboxes.append(loss_dict.get('loss_bbox', torch.tensor(0)).item())
        loss_gious.append(loss_dict.get('loss_giou', torch.tensor(0)).item())
        
        if step % 10 == 0 or step == 99:
            print(f"Step {step:3d}: Loss={loss.item():.4f} "
                  f"CE={loss_ces[-1]:.4f} "
                  f"BBox={loss_bboxes[-1]:.4f} "
                  f"IoU={loss_gious[-1]:.4f}")
    
    # 分析损失趋势
    print(f"\n{'='*60}")
    print(f"损失分析:")
    print(f"  - 初始 Loss: {losses[0]:.4f}")
    print(f"  - 最终 Loss: {losses[-1]:.4f}")
    print(f"  - Loss 下降: {losses[0] - losses[-1]:.4f}")
    
    # 预测分析
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
    
    pred_logits = outputs['pred_logits']
    pred_boxes = outputs['pred_boxes']
    
    scores = torch.sigmoid(pred_logits)
    max_scores, _ = scores.max(dim=-1)
    
    print(f"\n预测统计 (训练后):")
    print(f"  - 预测分数范围: [{max_scores.min().item():.4f}, {max_scores.max().item():.4f}]")
    print(f"  - 预测分数均值: {max_scores.mean().item():.4f}")
    
    print(f"  - 预测框 cx 范围: [{pred_boxes[:, :, 0].min().item():.4f}, {pred_boxes[:, :, 0].max().item():.4f}]")
    print(f"  - 预测框 cy 范围: [{pred_boxes[:, :, 1].min().item():.4f}, {pred_boxes[:, :, 1].max().item():.4f}]")
    print(f"  - 预测框 w 范围: [{pred_boxes[:, :, 2].min().item():.4f}, {pred_boxes[:, :, 2].max().item():.4f}]")
    print(f"  - 预测框 h 范围: [{pred_boxes[:, :, 3].min().item():.4f}, {pred_boxes[:, :, 3].max().item():.4f}]")
    
    # 检查是否能预测到GT附近
    if 'targets' in inputs:
        print(f"\nGT统计:")
        for i, target in enumerate(inputs['targets']):
            gt_boxes = target['boxes']
            print(f"  样本{i}: GT boxes = {gt_boxes}")
    
    # 计算最终指标
    from groundingdino.models.utils import box_cxcywh_to_xyxy, box_iou
    
    with torch.no_grad():
        pred_boxes_final = outputs['pred_boxes']  # [B, N, 4]
        pred_scores_final = torch.sigmoid(outputs['pred_logits']).max(dim=-1)[0]  # [B, N]
        
        print(f"\n最终预测详情:")
        for b in range(pred_boxes_final.shape[0]):
            best_idx = pred_scores_final[b].argmax()
            best_box = pred_boxes_final[b][best_idx]
            best_score = pred_scores_final[b][best_idx]
            print(f"  样本{b}: 最佳预测 box={best_box.tolist()}, score={best_score.item():.4f}")
    
    # 结论
    print(f"\n{'='*60}")
    if losses[-1] < losses[0] * 0.8:
        print(f"✅ 模型能在单批上收敛 (Loss 下降超过 20%)")
        print(f"   这说明模型基本工作正常，可以开始完整训练")
    elif losses[-1] < losses[0]:
        print(f"⚠️  模型在单批上有收敛但不稳定")
        print(f"   建议检查学习率和模型初始化")
    else:
        print(f"❌ 模型在单批上无法收敛!")
        print(f"   这说明存在严重的代码bug，需要检查:")
        print(f"   1. 损失函数计算")
        print(f"   2. 数据加载和预处理")
        print(f"   3. 模型前向传播")


if __name__ == "__main__":
    test_single_batch_overfit()
