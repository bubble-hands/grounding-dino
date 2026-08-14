"""
快速分析脚本 - 检查训练数据和预测
"""
import os, sys, json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groundingdino.config.GroundingDINO_Fused_Train import get_cfg
from groundingdino.datasets.dataset import MultiModalDataset
from groundingdino.models.groundingdino import GroundingDINO


def main():
    cfg = get_cfg()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("GroundingDINO 训练问题诊断")
    print("=" * 60)
    
    # 1. 数据检查
    print("\n【1】数据统计:")
    train_dataset = MultiModalDataset(cfg, split='train')
    val_dataset = MultiModalDataset(cfg, split='val')
    
    print(f"  - 训练集: {len(train_dataset)} 样本")
    print(f"  - 验证集: {len(val_dataset)} 样本")
    
    # 检查数据加载情况
    missing_count = 0
    for i in range(min(50, len(train_dataset))):
        item = train_dataset.data[i]
        for mod in ['rgb', 'ir', 'depth']:
            path = item.get(mod, '')
            if path and not os.path.exists(path):
                missing_count += 1
                break
    print(f"  - 前50样本中缺失图像: {missing_count}")
    
    # 2. 模型检查
    print("\n【2】模型检查:")
    model = GroundingDINO(cfg).to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")
    
    # 3. 预测检查
    print("\n【3】预测分析 (未训练模型):")
    train_dataset2 = MultiModalDataset(cfg, split='train')
    
    all_preds = []
    for i in range(min(10, len(train_dataset2))):
        inputs = train_dataset2[i]
        
        batch = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0).to(device)
            elif k == 'targets':
                batch[k] = [v]
            else:
                batch[k] = v
        
        with torch.no_grad():
            outputs = model(batch)
        
        pred_boxes = outputs['pred_boxes'][0].cpu().numpy()
        pred_scores = torch.sigmoid(outputs['pred_logits'][0]).max(dim=-1)[0].cpu().numpy()
        
        all_preds.append({
            'boxes': pred_boxes,
            'scores': pred_scores
        })
    
    # 统计预测分布
    all_boxes = np.concatenate([p['boxes'] for p in all_preds])
    all_scores = np.concatenate([p['scores'] for p in all_preds])
    
    print(f"  - 预测分数范围: [{all_scores.min():.4f}, {all_scores.max():.4f}]")
    print(f"  - 预测分数均值: {all_scores.mean():.4f}")
    print(f"  - 预测 cx 均值: {all_boxes[:, 0].mean():.4f}")
    print(f"  - 预测 cy 均值: {all_boxes[:, 1].mean():.4f}")
    print(f"  - 预测 w 均值: {all_boxes[:, 2].mean():.4f}")
    print(f"  - 预测 h 均值: {all_boxes[:, 3].mean():.4f}")
    
    # 检查预测是否集中
    std_cx = all_boxes[:, 0].std()
    std_cy = all_boxes[:, 1].std()
    print(f"  - cx 标准差: {std_cx:.4f}")
    print(f"  - cy 标准差: {std_cy:.4f}")
    
    if std_cx < 0.01 and std_cy < 0.01:
        print("  ⚠️ 预测框集中在一个小区域！模型未能学习位置信息")
    
    # 4. 损失测试
    print("\n【4】单批损失测试:")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    from torch.utils.data import DataLoader
    collator = __import__('groundingdino.datasets.dataset', fromlist=['MultiModalCollator']).MultiModalCollator(cfg)
    loader = DataLoader(train_dataset2, batch_size=4, shuffle=False, collate_fn=collator)
    
    batch = next(iter(loader))
    inputs = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
        else:
            inputs[k] = v
    
    losses = []
    for step in range(20):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    print(f"  - 初始 Loss: {losses[0]:.4f}")
    print(f"  - 最终 Loss: {losses[-1]:.4f}")
    print(f"  - Loss 变化: {losses[0] - losses[-1]:.4f}")
    
    if losses[-1] < losses[0]:
        print("  ✅ 模型能正常训练 (Loss 下降)")
    else:
        print("  ❌ 模型无法收敛 (Loss 未下降)")
    
    # 5. 总结和建议
    print("\n" + "=" * 60)
    print("【问题总结和建议】")
    print("=" * 60)
    print("""
问题: Val Loss 升高且 mAP 为 0

根本原因分析:
1. 过拟合: 训练 Loss 下降但 Val Loss 上升，说明模型记住了训练集噪声
2. 预训练缺失: BERT 和 Swin 都使用随机初始化，缺乏先验知识
3. 数据增强缺失: 之前代码中 _augment() 方法已定义但从未在 __getitem__ 中调用
4. 模型容量过大: 在 2875 样本上训练数百万参数的模型，容易过拟合
5. 学习率调度: 原配置使用固定学习率，缺少有效的 warmup 和 decay

已实施的修复:
1. ✅ 添加数据增强 (随机翻转、亮度/对比度调整) 并正确应用到 __getitem__
2. ✅ 减小模型容量 (decoder 6→2层, hidden_dim 128→64, ffn 2048→512)
3. ✅ 降低学习率 (5e-5→2e-5)，增加 weight_decay (1e-3)
4. ✅ 增加训练轮数 (50→80) 和 batch_size (4→8)
5. ✅ 延长 warmup (3→5 epochs)

建议:
1. 重新启动训练，使用修复后的配置
2. 监控训练曲线，如果 Val Loss 再次上升则提前停止
3. 考虑增加数据量或使用数据增强
4. 如果条件允许，下载预训练权重初始化 BERT 和 Swin
""")


if __name__ == "__main__":
    main()
