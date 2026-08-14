"""
深度诊断脚本 - 分析 Val Loss 升高且 mAP 为 0 的根本原因
"""
import os
import sys
import json
import torch
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groundingdino.config.GroundingDINO_Fused_Train import get_cfg
from groundingdino.datasets.dataset import MultiModalDataset
from groundingdino.models.groundingdino import GroundingDINO


def check_data_quality():
    """检查数据质量"""
    print("=" * 60)
    print("【1】数据质量检查")
    print("=" * 60)
    
    cfg = get_cfg()
    
    # 加载训练集和验证集
    train_dataset = MultiModalDataset(cfg, split='train')
    val_dataset = MultiModalDataset(cfg, split='val')
    
    print(f"\n训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 分析标注框分布
    for split, dataset in [('train', train_dataset), ('val', val_dataset)]:
        all_boxes = []
        all_labels = []
        img_sizes = []
        missing_images = 0
        
        for i in range(min(100, len(dataset))):
            item = dataset.data[i]
            
            # 检查图像是否存在
            for mod in ['rgb', 'ir', 'depth']:
                img_path = item.get(mod)
                if img_path:
                    if not os.path.isabs(img_path):
                        img_path = os.path.join(cfg.DATASETS.DATA_PATH, img_path)
                    if not os.path.exists(img_path):
                        missing_images += 1
                        break
            
            # 检查标注
            if 'annotations' in item and len(item['annotations']) > 0:
                for ann in item['annotations']:
                    bbox = ann['bbox']
                    all_boxes.append(bbox)
                    all_labels.append(ann['category_id'])
                    if 'img_size' in ann:
                        img_sizes.append(ann['img_size'])
        
        print(f"\n【{split}】前100样本统计:")
        print(f"  - 缺失图像数: {missing_images}")
        print(f"  - 标注框数: {len(all_boxes)}")
        
        if len(all_boxes) > 0:
            boxes_arr = np.array(all_boxes)
            
            # 检查坐标范围
            x_vals = boxes_arr[:, 0]
            y_vals = boxes_arr[:, 1]
            w_vals = boxes_arr[:, 2]
            h_vals = boxes_arr[:, 3]
            
            print(f"  - x 范围: [{x_vals.min():.1f}, {x_vals.max():.1f}]")
            print(f"  - y 范围: [{y_vals.min():.1f}, {y_vals.max():.1f}]")
            print(f"  - w 范围: [{w_vals.min():.1f}, {w_vals.max():.1f}]")
            print(f"  - h 范围: [{h_vals.min():.1f}, {h_vals.max():.1f}]")
            
            # 检查是否为像素坐标
            if len(img_sizes) > 0:
                sizes_arr = np.array(img_sizes)
                print(f"  - 图像尺寸范围: w[{sizes_arr[:, 0].min()}-{sizes_arr[:, 0].max()}], h[{sizes_arr[:, 1].min()}-{sizes_arr[:, 1].max()}]")
                
                # 计算归一化后的坐标
                norm_cx = (x_vals + w_vals / 2) / sizes_arr[0, 0]
                norm_cy = (y_vals + h_vals / 2) / sizes_arr[0, 1]
                norm_w = w_vals / sizes_arr[0, 0]
                norm_h = h_vals / sizes_arr[0, 1]
                
                print(f"  - 归一化 cx 范围: [{norm_cx.min():.4f}, {norm_cx.max():.4f}]")
                print(f"  - 归一化 cy 范围: [{norm_cy.min():.4f}, {norm_cy.max():.4f}]")
                print(f"  - 归一化 w 范围: [{norm_w.min():.4f}, {norm_w.max():.4f}]")
                print(f"  - 归一化 h 范围: [{norm_h.min():.4f}, {norm_h.max():.4f}]")
                
                # 检查是否超出 [0,1]
                out_of_range = np.sum((norm_cx < 0) | (norm_cx > 1) | (norm_cy < 0) | (norm_cy > 1) | 
                                      (norm_w < 0) | (norm_w > 1) | (norm_h < 0) | (norm_h > 1))
                print(f"  - 超出 [0,1] 范围的框: {out_of_range}/{len(all_boxes)}")
            
            # 类别分布
            label_counts = Counter(all_labels)
            print(f"  - 类别分布: {dict(label_counts)}")
        
        # 检查文本内容
        texts = [item.get('text', '') for item in dataset.data[:50]]
        print(f"  - 示例文本:")
        for t in texts[:3]:
            print(f"    '{t}'")
    
    return train_dataset, val_dataset


def check_model_predictions():
    """检查模型预测输出"""
    print("\n" + "=" * 60)
    print("【2】模型预测输出检查")
    print("=" * 60)
    
    cfg = get_cfg()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 创建模型
    model = GroundingDINO(cfg).to(device)
    model.eval()
    
    # 检查输出统计
    train_dataset = MultiModalDataset(cfg, split='train')
    
    all_pred_scores = []
    all_pred_boxes = []
    all_gt_boxes = []
    all_gt_labels = []
    
    num_samples = min(20, len(train_dataset))
    
    for i in range(num_samples):
        inputs = train_dataset[i]
        
        # 移到设备
        batch_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                batch_inputs[k] = v.unsqueeze(0).to(device)
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                batch_inputs[k] = v
            else:
                batch_inputs[k] = v
        
        with torch.no_grad():
            outputs = model(batch_inputs)
        
        # 分析预测
        pred_logits = outputs['pred_logits']  # [1, N, 2]
        pred_boxes = outputs['pred_boxes']    # [1, N, 4]
        
        scores = torch.sigmoid(pred_logits).squeeze(0)  # [N, 2]
        max_scores, _ = scores.max(dim=-1)  # [N]
        
        all_pred_scores.extend(max_scores.cpu().numpy().tolist())
        all_pred_boxes.extend(pred_boxes.squeeze(0).cpu().numpy().tolist())
        
        # GT
        if 'targets' in inputs:
            target = inputs['targets']
            all_gt_boxes.extend(target['boxes'].numpy().tolist())
            all_gt_labels.extend(target['labels'].numpy().tolist())
    
    print(f"\n预测统计 (前{num_samples}样本):")
    print(f"  - 预测框数量: {len(all_pred_boxes)}")
    
    if len(all_pred_scores) > 0:
        scores_arr = np.array(all_pred_scores)
        print(f"  - 预测分数范围: [{scores_arr.min():.4f}, {scores_arr.max():.4f}]")
        print(f"  - 预测分数均值: {scores_arr.mean():.4f}")
        print(f"  - 预测分数标准差: {scores_arr.std():.4f}")
        
        # 分数分布
        bins = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
        hist, _ = np.histogram(scores_arr, bins=bins)
        print(f"  - 分数分布: {dict(zip([f'{b1}-{b2}' for b1, b2 in zip(bins[:-1], bins[1:])], hist))}")
    
    if len(all_pred_boxes) > 0:
        boxes_arr = np.array(all_pred_boxes)
        print(f"  - 预测 cx 范围: [{boxes_arr[:, 0].min():.4f}, {boxes_arr[:, 0].max():.4f}]")
        print(f"  - 预测 cy 范围: [{boxes_arr[:, 1].min():.4f}, {boxes_arr[:, 1].max():.4f}]")
        print(f"  - 预测 w 范围: [{boxes_arr[:, 2].min():.4f}, {boxes_arr[:, 2].max():.4f}]")
        print(f"  - 预测 h 范围: [{boxes_arr[:, 3].min():.4f}, {boxes_arr[:, 3].max():.4f}]")
        
        # 检查是否集中在某个区域
        mean_cx = boxes_arr[:, 0].mean()
        mean_cy = boxes_arr[:, 1].mean()
        print(f"  - 预测中心均值: cx={mean_cx:.4f}, cy={mean_cy:.4f}")
    
    if len(all_gt_boxes) > 0:
        gt_arr = np.array(all_gt_boxes)
        print(f"\nGT统计:")
        print(f"  - GT框数量: {len(all_gt_boxes)}")
        print(f"  - GT cx 范围: [{gt_arr[:, 0].min():.4f}, {gt_arr[:, 0].max():.4f}]")
        print(f"  - GT cy 范围: [{gt_arr[:, 1].min():.4f}, {gt_arr[:, 1].max():.4f}]")
        print(f"  - GT w 范围: [{gt_arr[:, 2].min():.4f}, {gt_arr[:, 2].max():.4f}]")
        print(f"  - GT h 范围: [{gt_arr[:, 3].min():.4f}, {gt_arr[:, 3].max():.4f}]")
        
        # 计算预测和GT的IoU
        from groundingdino.models.utils import box_cxcywh_to_xyxy, box_iou
        
        pred_xyxy = torch.from_numpy(boxes_arr) if len(boxes_arr) > 0 else torch.zeros(0, 4)
        gt_xyxy = torch.from_numpy(gt_arr)
        
        if len(pred_xyxy) > 0 and len(gt_xyxy) > 0:
            # 取每个样本的最高分预测
            pred_xyxy_img = pred_xyxy[:num_samples]  # 简化：取前N个
            
            ious, _ = box_iou(pred_xyxy_img[:len(gt_xyxy)], gt_xyxy)
            max_ious = ious.max(dim=0)[0]  # 每个GT的最佳IoU
            
            print(f"\n预测与GT的IoU分析:")
            print(f"  - 最佳IoU均值: {max_ious.mean():.4f}")
            print(f"  - IoU>0.5的GT比例: {(max_ious > 0.5).float().mean():.4f}")
            print(f"  - IoU>0.3的GT比例: {(max_ious > 0.3).float().mean():.4f}")
            print(f"  - IoU>0.1的GT比例: {(max_ious > 0.1).float().mean():.4f}")


def check_loss_computation():
    """检查损失函数计算"""
    print("\n" + "=" * 60)
    print("【3】损失函数检查")
    print("=" * 60)
    
    cfg = get_cfg()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = GroundingDINO(cfg).to(device)
    model.train()
    
    train_dataset = MultiModalDataset(cfg, split='train')
    
    # 取一个样本
    inputs = train_dataset[0]
    
    batch_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            batch_inputs[k] = v.unsqueeze(0).to(device)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            batch_inputs[k] = v
        else:
            batch_inputs[k] = v
    
    # 前向传播
    outputs = model(batch_inputs)
    
    print(f"\n损失组成:")
    loss_dict = outputs.get('loss_dict', {})
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"  - {k}: {v.item():.4f}")
    
    total_loss = outputs.get('loss', None)
    if total_loss is not None:
        print(f"  - 总损失: {total_loss.item():.4f}")
    
    # 检查预测和GT的匹配情况
    with torch.no_grad():
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        print(f"\n预测统计:")
        print(f"  - pred_logits shape: {pred_logits.shape}")
        print(f"  - pred_logits 范围: [{pred_logits.min().item():.4f}, {pred_logits.max().item():.4f}]")
        print(f"  - pred_logits 均值: {pred_logits.mean().item():.4f}")
        
        scores = torch.sigmoid(pred_logits)
        print(f"  - sigmoid(scores) 范围: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
        print(f"  - sigmoid(scores) 均值: {scores.mean().item():.4f}")
        
        print(f"  - pred_boxes shape: {pred_boxes.shape}")
        print(f"  - pred_boxes 范围: [{pred_boxes.min().item():.4f}, {pred_boxes.max().item():.4f}]")
        print(f"  - pred_boxes 均值: {pred_boxes.mean().item():.4f}")
        
        # 检查GT
        if 'targets' in inputs:
            target = inputs['targets']
            gt_boxes = target['boxes']
            gt_labels = target['labels']
            
            print(f"\nGT统计:")
            print(f"  - gt_boxes: {gt_boxes}")
            print(f"  - gt_labels: {gt_labels}")
            
            # 使用匈牙利匹配
            from groundingdino.models.losses import HungarianMatcher
            matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
            
            targets = [{'labels': gt_labels, 'boxes': gt_boxes}]
            indices = matcher(outputs, targets)
            
            print(f"\n匈牙利匹配结果:")
            print(f"  - 匹配索引: {indices}")
            
            # 检查匹配后的预测
            if len(indices) > 0 and len(indices[0]) > 0:
                src_idx, tgt_idx = indices[0]
                matched_pred_boxes = pred_boxes[0][src_idx]
                matched_gt_boxes = gt_boxes[tgt_idx]
                
                print(f"  - 匹配的预测框: {matched_pred_boxes}")
                print(f"  - 匹配的GT框: {matched_gt_boxes}")
                
                # 计算匹配框的IoU
                from groundingdino.models.utils import box_cxcywh_to_xyxy, generalized_box_iou
                
                pred_xyxy = box_cxcywh_to_xyxy(matched_pred_boxes)
                gt_xyxy = box_cxcywh_to_xyxy(matched_gt_boxes)
                
                ious = torch.diag(generalized_box_iou(pred_xyxy, gt_xyxy))
                print(f"  - 匹配框的GIoU: {ious}")


def analyze_training_logs():
    """分析训练日志"""
    print("\n" + "=" * 60)
    print("【4】训练日志分析")
    print("=" * 60)
    
    log_dir = "g:/Grounding DINO/logs"
    
    # 检查epoch指标
    epoch_csv = os.path.join(log_dir, "metrics_epoch.csv")
    if os.path.exists(epoch_csv):
        import csv
        
        with open(epoch_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        print(f"\nEpoch级别指标:")
        print(f"{'Epoch':<8}{'Train Loss':<12}{'Val Loss':<12}{'mAP50':<12}{'LR':<15}")
        print("-" * 60)
        
        for row in rows:
            epoch = int(row['epoch'])
            train_loss = float(row['train_loss'])
            val_loss = float(row['val_loss']) if row['val_loss'] else None
            map50 = float(row['mAP50']) if row['mAP50'] else None
            lr = row['lr']
            
            val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
            map50_str = f"{map50:.6f}" if map50 is not None else "N/A"
            
            print(f"{epoch:<8}{train_loss:<12.4f}{val_loss_str:<12}{map50_str:<12}{lr:<15}")
        
        # 趋势分析
        if len(rows) >= 2:
            first = rows[0]
            last = rows[-1]
            
            train_loss_change = float(last['train_loss']) - float(first['train_loss'])
            val_loss_change = float(last['val_loss']) - float(first['val_loss']) if first['val_loss'] and last['val_loss'] else None
            
            print(f"\n趋势分析:")
            print(f"  - Train Loss变化: {train_loss_change:+.4f} ({'下降' if train_loss_change < 0 else '上升'})")
            if val_loss_change is not None:
                print(f"  - Val Loss变化: {val_loss_change:+.4f} ({'下降' if val_loss_change < 0 else '上升'})")
            
            # 检查是否过拟合
            if train_loss_change < 0 and val_loss_change is not None and val_loss_change > 0:
                print(f"  - ⚠️ 检测到过拟合: Train Loss下降但Val Loss上升")
            
            # 检查mAP
            map_values = [float(row['mAP50']) for row in rows if row['mAP50']]
            if map_values:
                print(f"  - mAP50 最大值: {max(map_values):.6f}")
                print(f"  - mAP50 最小值: {min(map_values):.6f}")
                print(f"  - mAP50 均值: {sum(map_values)/len(map_values):.6f}")
                
                if max(map_values) < 0.01:
                    print(f"  - ⚠️ mAP50 始终接近0，模型未能学到有效的检测能力")
    
    # 检查batch级别损失变化
    batch_csv = os.path.join(log_dir, "metrics_batch.csv")
    if os.path.exists(batch_csv):
        import csv
        
        with open(batch_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if rows:
            print(f"\nBatch级别损失统计 (共{len(rows)}条):")
            
            # 按epoch分组
            epoch_losses = {}
            for row in rows:
                epoch = int(row['epoch'])
                loss = float(row['loss'])
                if epoch not in epoch_losses:
                    epoch_losses[epoch] = []
                epoch_losses[epoch].append(loss)
            
            for epoch in sorted(epoch_losses.keys()):
                losses = epoch_losses[epoch]
                print(f"  Epoch {epoch}: Loss均值={np.mean(losses):.4f}, 标准差={np.std(losses):.4f}, 范围=[{min(losses):.4f}, {max(losses):.4f}]")


def main():
    print("=" * 70)
    print("  GroundingDINO 深度诊断 - Val Loss升高 & mAP为0")
    print("=" * 70)
    
    try:
        # 1. 数据质量检查
        train_dataset, val_dataset = check_data_quality()
        
        # 2. 模型预测输出检查
        check_model_predictions()
        
        # 3. 损失函数检查
        check_loss_computation()
        
        # 4. 训练日志分析
        analyze_training_logs()
        
        print("\n" + "=" * 70)
        print("诊断完成！")
        print("=" * 70)
        
        # 总结与建议
        print("\n【总结与建议】")
        print("1. 如果预测框集中在某个区域（如左下方），说明模型未能学习到空间位置信息")
        print("2. 如果mAP始终接近0，说明模型的分类/回归头未能有效工作")
        print("3. 如果Val Loss > Train Loss，说明存在过拟合")
        print("4. 建议检查：")
        print("   - 数据标注是否正确（坐标归一化）")
        print("   - 模型架构是否适合当前任务（参数量、层数）")
        print("   - 学习率和batch size是否合适")
        print("   - 是否使用了预训练权重")
        print("   - 损失函数权重是否平衡")
        
    except Exception as e:
        print(f"\n❌ 诊断出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
