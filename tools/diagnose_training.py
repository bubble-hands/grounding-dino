"""
全面检查训练数据质量和模型训练问题诊断。
"""
import os
import sys
import json
import numpy as np
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def check_data_quality():
    """检查训练和验证数据质量"""
    print("="*80)
    print("  训练数据质量全面检查")
    print("="*80)
    
    for split in ['train', 'val']:
        data_path = os.path.join(PROJECT_ROOT, 'data', f'{split}.json')
        print(f"\n{'='*40}")
        print(f"  {split.upper()} 数据集")
        print(f"{'='*40}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"样本总数: {len(data)}")
        
        # 1. 检查图像存在性
        missing_images = 0
        corrupted_images = 0
        valid_samples = 0
        
        for item in data:
            rgb_path = item.get('rgb', '')
            if not os.path.exists(rgb_path):
                missing_images += 1
                continue
            
            # 尝试加载图像
            try:
                from PIL import Image
                img = Image.open(rgb_path)
                img.verify()
                valid_samples += 1
            except:
                corrupted_images += 1
                missing_images += 1
        
        print(f"有效样本: {valid_samples}/{len(data)} ({valid_samples/len(data)*100:.1f}%)")
        print(f"缺失/损坏图像: {missing_images}")
        
        # 2. 检查标注分布
        total_annotations = 0
        category_dist = Counter()
        bbox_sizes = []
        aspect_ratios = []
        center_positions = []
        invalid_bboxes = []
        
        for item in data:
            for ann in item.get('annotations', []):
                total_annotations += 1
                cat_id = ann.get('category_id', -1)
                category_dist[cat_id] += 1
                
                bbox = ann.get('bbox', [0, 0, 0, 0])
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    img_size = ann.get('img_size', [1920, 1080])
                    W, H = img_size
                    
                    # 检查 bbox 是否有效
                    if w <= 0 or h <= 0:
                        invalid_bboxes.append((item.get('query_id', 'unknown'), bbox))
                    elif x + w > W or y + h > H:
                        invalid_bboxes.append((item.get('query_id', 'unknown'), bbox, f'超出图像范围 {W}x{H}'))
                    else:
                        # 计算归一化坐标
                        cx = (x + w/2) / W
                        cy = (y + h/2) / H
                        nw = w / W
                        nh = h / H
                        
                        center_positions.append((cx, cy))
                        bbox_sizes.append((nw, nh))
                        if h > 0:
                            aspect_ratios.append(w / h)
        
        print(f"\n标注总数: {total_annotations}")
        print(f"类别分布: {dict(category_dist)}")
        print(f"无效 bbox: {len(invalid_bboxes)}")
        if invalid_bboxes:
            print(f"  示例: {invalid_bboxes[:3]}")
        
        # 3. 检查 bbox 尺寸分布 (归一化)
        if bbox_sizes:
            sizes = np.array(bbox_sizes)
            print(f"\nbbox 尺寸分布 (归一化):")
            print(f"  宽度 w: min={sizes[:,0].min():.4f}, max={sizes[:,0].max():.4f}, mean={sizes[:,0].mean():.4f}")
            print(f"  高度 h: min={sizes[:,1].min():.4f}, max={sizes[:,1].max():.4f}, mean={sizes[:,1].mean():.4f}")
            print(f"  面积:   min={(sizes[:,0]*sizes[:,1]).min():.6f}, max={(sizes[:,0]*sizes[:,1]).max():.6f}, mean={(sizes[:,0]*sizes[:,1]).mean():.6f}")
            
            # 检查目标大小比例
            small_targets = sum(1 for s in sizes if s[0] * s[1] < 0.01)
            medium_targets = sum(1 for s in sizes if 0.01 <= s[0] * s[1] < 0.1)
            large_targets = sum(1 for s in sizes if s[0] * s[1] >= 0.1)
            print(f"  目标大小分布: 小(<1%)={small_targets}, 中(1-10%)={medium_targets}, 大(>10%)={large_targets}")
        
        # 4. 检查中心点位置分布
        if center_positions:
            centers = np.array(center_positions)
            print(f"\n中心点位置分布 (归一化):")
            print(f"  cx: min={centers[:,0].min():.4f}, max={centers[:,0].max():.4f}, mean={centers[:,0].mean():.4f}")
            print(f"  cy: min={centers[:,1].min():.4f}, max={centers[:,1].max():.4f}, mean={centers[:,1].mean():.4f}")
            
            # 象限分布
            cx = centers[:, 0]
            cy = centers[:, 1]
            left = cx < 0.5
            right = cx >= 0.5
            top = cy < 0.5
            bottom = cy >= 0.5
            
            print(f"\n  象限分布:")
            print(f"    左下 (cx<0.5, cy>=0.5): {(left & bottom).sum()} ({(left & bottom).mean()*100:.1f}%)")
            print(f"    左上 (cx<0.5, cy<0.5): {(left & top).sum()} ({(left & top).mean()*100:.1f}%)")
            print(f"    右上 (cx>=0.5, cy<0.5): {(right & top).sum()} ({(right & top).mean()*100:.1f}%)")
            print(f"    右下 (cx>=0.5, cy>=0.5): {(right & bottom).sum()} ({(right & bottom).mean()*100:.1f}%)")
            
            # 计算标准差
            print(f"\n  cx std: {centers[:,0].std():.4f}")
            print(f"  cy std: {centers[:,1].std():.4f}")
            
            # 检查分布是否过于集中
            if centers[:,0].std() < 0.1 or centers[:,1].std() < 0.1:
                print("  ⚠️ 警告: 中心点分布过于集中，可能影响模型学习!")
        
        # 5. 检查宽高比分布
        if aspect_ratios:
            ar = np.array(aspect_ratios)
            print(f"\n宽高比分布: min={ar.min():.2f}, max={ar.max():.2f}, mean={ar.mean():.2f}, median={np.median(ar):.2f}")
            
            # 计算 log 宽高比的分布
            log_ar = np.log(ar)
            print(f"log(宽高比) std: {log_ar.std():.4f}")
            
            if log_ar.std() > 1.0:
                print("  ⚠️ 警告: 宽高比变化较大，可能增加训练难度!")

def check_training_logs():
    """检查训练日志和指标趋势"""
    print("\n" + "="*80)
    print("  训练日志分析")
    print("="*80)
    
    # 读取 epoch 级指标
    metrics_path = os.path.join(PROJECT_ROOT, 'logs', 'metrics_epoch.csv')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            lines = f.readlines()
        
        print(f"\nEpoch 级指标 (共 {len(lines)-1} 条):")
        print("-"*100)
        print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>10} | {'mAP@50':>10} | {'LR':>14} | {'耗时(s)':>10}")
        print("-"*100)
        
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                epoch = parts[1]
                train_loss = float(parts[2]) if parts[2] != '' else float('nan')
                val_loss = float(parts[3]) if parts[3] != '' else float('nan')
                map50 = float(parts[4]) if parts[4] != '' else float('nan')
                lr = float(parts[5]) if parts[5] != '' else float('nan')
                elapsed = float(parts[6]) if parts[6] != '' else float('nan')
                
                loss_diff = val_loss - train_loss if not np.isnan(val_loss) and not np.isnan(train_loss) else float('nan')
                warning = " ⚠️" if loss_diff > 0.5 else ""
                
                print(f"{epoch:>6} | {train_loss:>12.4f} | {val_loss:>10.4f} | {map50:>10.6f} | {lr:>14.2e} | {elapsed:>10.1f}{warning}")
        
        # 分析趋势
        if len(lines) > 2:
            train_losses = []
            val_losses = []
            map50s = []
            
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    if parts[2]:
                        train_losses.append(float(parts[2]))
                    if parts[3]:
                        val_losses.append(float(parts[3]))
                    if parts[4]:
                        map50s.append(float(parts[4]))
            
            if train_losses:
                print(f"\n--- 训练趋势分析 ---")
                print(f"Train Loss: 第1轮={train_losses[0]:.4f}, 最后一轮={train_losses[-1]:.4f}")
                print(f"  下降: {train_losses[0] - train_losses[-1]:.4f} ({(train_losses[0] - train_losses[-1])/train_losses[0]*100:.1f}%)")
            
            if val_losses:
                print(f"Val Loss: 第1轮={val_losses[0]:.4f}, 最后一轮={val_losses[-1]:.4f}")
                if len(val_losses) >= 2:
                    val_trend = np.diff(val_losses)
                    val_increase = sum(1 for x in val_trend if x > 0)
                    val_decrease = sum(1 for x in val_trend if x < 0)
                    print(f"  Val Loss 上升次数: {val_increase}, 下降次数: {val_decrease}")
                
                # 检查过拟合
                if len(val_losses) >= 2 and val_losses[-1] > val_losses[0]:
                    print("  ⚠️ 警告: Val Loss 不降反升，可能存在过拟合!")
                    if train_losses and train_losses[-1] < train_losses[0]:
                        print("  📈 过拟合信号: Train Loss 下降但 Val Loss 上升")
            
            if map50s:
                print(f"\nmAP@50: 第1轮={map50s[0]:.6f}, 最后一轮={map50s[-1]:.6f}")
                if max(map50s) < 0.01:
                    print("  ⚠️ 警告: mAP 始终接近 0，模型没有学到有效特征!")
    else:
        print("\n未找到 metrics_epoch.csv")
    
    # 读取 batch 级指标（前100条和后100条对比）
    batch_path = os.path.join(PROJECT_ROOT, 'logs', 'metrics_batch.csv')
    if os.path.exists(batch_path):
        with open(batch_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) > 100:
            print(f"\n--- Batch 级指标分析 (对比首尾) ---")
            
            # 前100条 batch loss
            first_losses = []
            for line in lines[1:101]:
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    loss = float(parts[4]) if parts[4] else float('nan')
                    if not np.isnan(loss):
                        first_losses.append(loss)
            
            # 后100条 batch loss
            last_losses = []
            for line in lines[-100:]:
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    loss = float(parts[4]) if parts[4] else float('nan')
                    if not np.isnan(loss):
                        last_losses.append(loss)
            
            if first_losses and last_losses:
                print(f"前100 batch avg loss: {np.mean(first_losses):.4f}")
                print(f"后100 batch avg loss: {np.mean(last_losses):.4f}")
                
                # 检查 loss 方差
                if np.std(first_losses) > 2.0:
                    print("  ⚠️ 警告: 初期 loss 方差较大，可能存在训练不稳定!")

def main():
    check_data_quality()
    check_training_logs()
    
    print("\n" + "="*80)
    print("  诊断结论与建议")
    print("="*80)
    print("""
基于以上分析，如果 mAP 持续为 0，可能的原因包括：

1. 【高优先级】损失函数计算问题
   - 检查坐标格式是否一致（cxcywh vs xyxy）
   - 检查 GIoU 计算中是否包含面积惩罚项
   
2. 【高优先级】评估指标计算问题
   - 检查 mAP 计算时的坐标格式
   - 检查阈值设置是否合理
   
3. 【中优先级】训练策略问题
   - 学习率是否过大/过小
   - 损失权重分配是否合理
   - 是否需要更长的训练时间
   
4. 【中优先级】数据质量问题
   - 标注是否准确
   - 数据增强是否合理
   - 类别不平衡问题

建议下一步：检查 losses.py 和 metrics.py 的实现细节。
""")

if __name__ == '__main__':
    main()