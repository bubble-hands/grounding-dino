# Grounding DINO 多模态目标定位

基于 Grounding DINO 架构的多模态视觉理解与推理模型。以文本语义为引导，融合 **RGB可见光、红外(IR)、深度(Depth)** 三种视觉模态，学习从多源特征到目标空间位置的映射，实现跨场景的鲁棒目标定位。

## 项目结构

```
Grounding DINO/
├── groundingdino/                    # 核心模型库
│   ├── config/
│   │   ├── defaults.py               # 默认配置定义 (yacs CfgNode)
│   │   └── GroundingDINO_Fused_Train.py  # 训练配置文件 (v5 P0优化版)
│   ├── datasets/
│   │   └── dataset.py                # 多模态数据集 + 几何数据增强 + collator
│   ├── engine/
│   │   ├── trainer.py                # 训练器 (Warmup/Cosine LR/AMP/早停/暂停恢复)
│   │   └── metrics.py                # mAP@50 计算 + MetricsLogger (CSV/JSONL)
│   ├── inference/
│   │   └── predictor.py              # 推理器 (加载checkpoint + 预测 + 可视化)
│   ├── models/
│   │   ├── backbone.py               # Swin Transformer主干 + ModalityAdapter + 门控融合
│   │   ├── text_encoder.py           # BERT文本编码器 (支持HF镜像/本地缓存/随机初始化)
│   │   ├── feature_enhancer.py       # 跨模态特征增强 (Image↔Text双向交叉注意力)
│   │   ├── decoder.py                # Transformer解码器 (QueryInitializer + DecoderLayer)
│   │   ├── groundingdino.py          # 主模型 (前向传播 + 匈牙利匹配 + box_head bias初始化)
│   │   ├── losses.py                 # SetCriterion + HungarianMatcher + SigmoidFocalLoss + GIoU
│   │   ├── head.py                   # GroundingHead (分类头 + 回归头)
│   │   └── utils.py                  # box转换 + sigmoid_focal_loss + generalized_box_iou
│   └── version.py
├── tools/                            # 工具脚本
│   ├── train.py                      # 训练入口 (支持--resume断点续训)
│   ├── inference.py                  # 推理入口
│   ├── validate.py                   # 验证脚本
│   ├── monitor_training.py           # 实时监控loss/mAP曲线
│   ├── pause_training.py             # 训练暂停/恢复/停止/状态查询
│   ├── auto_pause_at_epoch.py        # 指定epoch完成后自动暂停
│   ├── build_dataset_from_share_annot.py  # 从share_annot构建训练数据集
│   ├── merge_share_annot.py          # 合并多个JSON标注文件
│   ├── clean_output.py               # 清理output/logs目录
│   ├── generate_report.py            # 生成可视化对比报告 (HTML)
│   ├── baseline_inference.py         # 裸跑基线推理
│   ├── run_trained_inference.py      # 使用训练模型推理
│   ├── analyze_data_distribution.py  # 数据分布分析工具
│   ├── diagnose_model_output.py      # 模型输出诊断
│   └── visualize_val.py              # 验证集结果可视化
├── data/                             # 数据集 (train.json / val.json)
├── output/                           # 训练输出 (checkpoint + 模型权重)
├── logs/                             # 训练日志 (training.log + metrics CSV)
├── test_results/                     # 测试集推理结果
├── 初赛数据集-基于大模型的多模态视觉理解与推理/  # 测试集 (Images/rgb+ir+depth)
├── requirements.txt
└── setup.py
```

## 模型架构

```
输入: RGB(3ch) + IR(1ch) + Depth(1ch) + 文本描述
                    │
    ┌───────────────┼───────────────┐
    │               │               │
  ModalityAdapter ModalityAdapter ModalityAdapter
    │ (3→96ch)     │ (1→96ch)     │ (1→96ch)
    └───────┬───────┘               │
            │                       │
     MultiModalFusion (门控加权)     │
            │                       │
     Swin Transformer主干            │
     ├ Stage1: 96ch                 │
     ├ Stage2: 192ch                │
     ├ Stage3: 384ch      BERT文本编码器 (bert-base-uncased)
     └ Stage4: 768ch        │ (768dim)
            │                │
     FeatureEnhancer ←──────┘
     ├ Image→Text交叉注意力 (每层)
     ├ Text→Image交叉注意力 (每层)
     └ Text自注意力
            │
     Transformer Decoder (2层)
     ├ QueryInitializer (50 queries, 文本相似度排序)
     ├ Self-Attention
     ├ Image Cross-Attention
     └ Text Cross-Attention
            │
     ┌──────┴──────┐
     │             │
  ClassHead      BoxHead
  (→1 score)    (→4 cxcywh)
```

### 关键设计

| 组件 | 说明 |
|------|------|
| **模态适配器** | 每个模态独立 Conv2D→BN→GELU，映射到统一通道空间 |
| **门控融合** | 对各模态特征做 Sigmoid 门控 + Softmax 加权求和 |
| **BERT 文本编码器** | 加载 `bert-base-uncased` 预训练权重，通过 HF 镜像自动下载 |
| **跨模态注意力** | 每个 Swin stage 的图像特征与 BERT 文本特征双向交叉注意力 |
| **Query 初始化** | 50 个可学习 query，按与文本 CLS token 的相似度排序选择 |
| **匈牙利匹配** | `cost_class=5, cost_bbox=8, cost_giou=6`，平衡分类与定位 |
| **损失函数** | SigmoidFocalLoss (分类) + L1 Bbox Loss + GIoU Loss |
| **box_head bias 初始化** | bias 设为 `[0, 0, -2.197, -1.386]`，初始预测 w≈0.1 h≈0.2（小目标先验） |

## 环境要求

- Python 3.8+
- PyTorch 2.0+ (CUDA 11.8+ / CUDA 12.x)
- NVIDIA GPU (建议 ≥ 8GB 显存)

## 安装

```bash
# 安装 PyTorch (按你的CUDA版本选择)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

## 数据准备

### 数据格式

每条 JSON 记录包含：

```json
{
  "query_id": "001_00000001",
  "rgb": "/path/to/Train-001/001/color/00000001.png",
  "ir": "/path/to/Train-001/001/infrared/00000001.png",
  "depth": "/path/to/Train-001/001/depth/00000001.png",
  "text": "The white umbrella on the left side",
  "annotations": [{
    "category_id": 0,
    "bbox": [x, y, w, h],
    "img_size": [1920, 1080]
  }]
}
```

> bbox 为绝对像素坐标，`dataset.py` 的 `_prepare_targets` 会用 `img_size` 归一化到 [0,1]。

### 从 share_annot 构建数据集

```bash
# 合并 share_annot 下的 train/val JSON
python tools/merge_share_annot.py

# 构建 data/train.json + data/val.json (80:20划分)
python tools/build_dataset_from_share_annot.py
```

## 训练

### 启动训练

```bash
python tools/train.py --config groundingdino/config/GroundingDINO_Fused_Train.py
```

### 从 checkpoint 恢复

```bash
python tools/train.py --config groundingdino/config/GroundingDINO_Fused_Train.py --resume output/latest_checkpoint.pth
```

### 训练配置 (v5 P0优化版)

| 参数 | 值 | 说明 |
|------|-----|------|
| `SOLVER.BASE_LR` | 8e-6 | 基础学习率 |
| `SOLVER.WEIGHT_DECAY` | 1e-3 | 权重衰减 |
| `SOLVER.EPOCHS` | 100 | 总训练轮数 |
| `SOLVER.BATCH_SIZE` | 8 | 批次大小 |
| `SOLVER.WARMUP_EPOCHS` | 8 | Warmup 轮数 (线性) |
| `SOLVER.AMP` | True | 混合精度训练 |
| `SOLVER.NUM_WORKERS` | 0 | 数据加载线程 |
| `MODEL.NUM_QUERIES` | 50 | 查询数量 |
| `MODEL.HIDDEN_DIM` | 64 | 隐藏层维度 |
| `MODEL.DECODER.NUM_LAYERS` | 2 | 解码器层数 |
| `TEST.EARLY_STOPPING_PATIENCE` | 25 | 早停容忍轮数 |

LR 调度：前 8 个 epoch 线性 Warmup → 之后 Cosine Annealing 衰减。

### 数据增强 (训练时)

| 增强方式 | 参数 | 说明 |
|---------|------|------|
| 随机水平翻转 | p=0.5 | 同步翻转标注框 |
| 随机垂直翻转 | p=0.3 | 同步翻转标注框 |
| 随机缩放 | scale∈[0.7, 1.3] | 改变目标尺寸和相对位置 |
| 随机平移 | shift∈[-0.15, 0.15] | 将目标移到图像任意区域 |
| 亮度调整 | factor∈[0.8, 1.2] | 仅 RGB/IR/Depth |
| 对比度调整 | factor∈[0.8, 1.2] | 仅 RGB |

## 训练监控与控制

### 实时监控

```bash
# 终端实时刷新 (每5秒)
python tools/monitor_training.py

# 生成 loss/mAP 曲线图
python tools/monitor_training.py --plot

# 单次输出
python tools/monitor_training.py --once
```

### 训练暂停/恢复/停止

```bash
# 当前epoch结束后暂停
python tools/pause_training.py pause

# 从暂停恢复
python tools/pause_training.py resume

# 当前epoch结束后停止
python tools/pause_training.py stop

# 查看状态
python tools/pause_training.py status
```

### 指定 epoch 自动暂停

```bash
# Epoch 8完成后自动暂停 (每20秒轮询日志)
python tools/auto_pause_at_epoch.py --epoch 8
```

## 推理

```bash
python tools/inference.py \
    --config groundingdino/config/GroundingDINO_Fused_Train.py \
    --checkpoint output/best_model.pth \
    --rgb path/to/rgb.png \
    --ir path/to/ir.png \
    --depth path/to/depth.png \
    --text "a person walking" \
    --threshold 0.5
```

### 批量测试集推理

```bash
# 裸跑基线
python tools/baseline_inference.py

# 使用训练模型推理
python tools/run_trained_inference.py
```

### 生成可视化报告

```bash
python tools/generate_report.py
```

输出 HTML 报告，展示预测框与真实标注的差异对比。

## 日志与输出

| 路径 | 内容 |
|------|------|
| `logs/training.log` | 训练主日志 (epoch/batch/loss/lr) |
| `logs/metrics_batch.csv` | Batch 级指标 (loss_ce/loss_bbox/loss_giou) |
| `logs/metrics_epoch.csv` | Epoch 级指标 (train_loss/val_loss/mAP50/lr) |
| `output/latest_checkpoint.pth` | 最新 checkpoint (含 optimizer/scheduler/epoch 状态) |
| `output/grounding_dino_multi_modal_epoch_N_loss_X.pth` | 按 epoch 保存的模型权重 |
| `output/pause.flag` | 暂停标志文件 |
| `output/stop.flag` | 停止标志文件 |
| `output/resume.flag` | 恢复标志文件 |

## 技术特性

- **多模态融合**：RGB + IR + Depth 通过模态适配器和门控机制融合
- **文本引导定位**：BERT 语义编码 + 跨模态注意力实现文本→位置映射
- **几何数据增强**：翻转/缩放/平移打破位置记忆，迫使模型学习特征→位置映射
- **混合精度训练**：AMP 加速训练，减少显存占用
- **Warmup + Cosine LR**：线性 Warmup 稳定初期，Cosine 衰减精细收敛
- **早停机制**：Val Loss 连续 N 个 epoch 不下降时自动停止
- **断点续训**：完整保存/恢复 optimizer、scheduler、AMP scaler、epoch 状态
- **暂停/恢复控制**：通过 flag 文件实现训练进程的非侵入式控制
- **实时监控**：CSV 日志 + matplotlib 可视化 loss/mAP 曲线
- **box_head 先验初始化**：bias 初始化为小目标尺寸，避免大方框局部最优

## 依赖

见 [requirements.txt](file:///g:/Grounding%20DINO/requirements.txt)，核心依赖：

- `torch >= 2.0.0`
- `transformers >= 4.30.0` (BERT 文本编码器 + 分词器)
- `timm >= 0.9.0` (Swin Transformer 主干)
- `opencv-python >= 4.8.0` (数据增强 + 可视化)
- `yacs >= 0.1.8` (配置系统)
- `scipy >= 1.10.0` (匈牙利匹配 linear_sum_assignment)

## 许可证

MIT License
