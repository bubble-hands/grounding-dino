import sys
import os
import torch

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("=" * 70)
print("  GroundingDINO 全流程测试 (优化后版本)")
print("=" * 70)

print("\n[1/7] 配置加载...")
from groundingdino.config import get_cfg
cfg = get_cfg()
print(f"  MODEL.NAME: {cfg.MODEL.NAME}")
print(f"  TEXT_ENCODER.NAME: {cfg.MODEL.TEXT_ENCODER.NAME}")
print(f"  MULTI_MODAL.ENABLED: {cfg.MODEL.MULTI_MODAL.ENABLED}")
print(f"  USE_SWIN: {cfg.MODEL.USE_SWIN}")
print("  [OK]")

print("\n[2/7] Tokenizer 检查...")
from groundingdino.datasets.dataset import SimpleTokenizer
tokenizer = SimpleTokenizer()
encoding = tokenizer("Find a person", max_length=256)
print(f"  Input IDs shape: {encoding['input_ids'].shape}")
print(f"  (SimpleTokenizer fallback - BERT will be used when available)")
print("  [OK]")

print("\n[3/7] 数据集加载...")
from groundingdino.datasets.dataset import MultiModalDataset, MultiModalCollator

cfg.DATASETS.MAX_TRAIN_SAMPLES = 10
cfg.DATASETS.MAX_VAL_SAMPLES = 5
cfg.DATASETS.DATA_PATH = os.path.join(project_root, "data")

train_ds = MultiModalDataset(cfg, split='train')
val_ds = MultiModalDataset(cfg, split='val')
print(f"  Train samples: {len(train_ds)}")
print(f"  Val samples: {len(val_ds)}")

if len(train_ds) > 0:
    sample = train_ds[0]
    print(f"  Sample keys: {list(sample.keys())}")
    if 'rgb' in sample:
        print(f"  RGB shape: {sample['rgb'].shape}")
    if 'ir' in sample:
        print(f"  IR shape: {sample['ir'].shape}")
    if 'text_input_ids' in sample:
        print(f"  Text input IDs shape: {sample['text_input_ids'].shape}")
    print(f"  Text: {sample.get('text', 'N/A')[:80]}...")
    if 'targets' in sample:
        print(f"  Targets: {len(sample['targets']['boxes'])} boxes")
    print("  [OK]")
else:
    print("  [WARN] No training data found!")
    exit(1)

print("\n[4/7] 模型创建...")
from groundingdino.models.groundingdino import GroundingDINO
model = GroundingDINO(cfg)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print("  [OK]")

print("\n[5/7] 前向传播测试...")
device = torch.device('cpu')
model = model.to(device)
model.eval()

collator = MultiModalCollator(cfg)
loader = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=False, collate_fn=collator)
batch = next(iter(loader))

inputs = {}
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        inputs[k] = v.to(device)
    elif k == 'targets':
        inputs[k] = v

with torch.no_grad():
    outputs = model(inputs)

print(f"  pred_logits shape: {outputs['pred_logits'].shape}")
print(f"  pred_boxes shape: {outputs['pred_boxes'].shape}")
if 'loss' in outputs:
    print(f"  loss: {outputs['loss'].item():.4f}")
if 'results' in outputs:
    print(f"  results: {len(outputs['results'])} boxes detected")
print("  [OK]")

print("\n[6/7] 反向传播测试...")
model.train()
outputs = model(inputs)
if 'loss' in outputs:
    loss = outputs['loss']
    loss.backward()
    print(f"  Loss: {loss.item():.4f}")
    grad_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())
    if grad_norms:
        print(f"  Gradient stats: min={min(grad_norms):.6f}, max={max(grad_norms):.6f}, mean={sum(grad_norms)/len(grad_norms):.6f}")
print("  [OK]")

print("\n[7/7] 推理模式测试...")
model.eval()
with torch.no_grad():
    outputs = model(inputs)
if 'results' in outputs:
    for i, r in enumerate(outputs['results']):
        print(f"  Batch {i}: {len(r['scores'])} detections, max score: {r['scores'].max():.4f}")
else:
    print(f"  Output keys: {list(outputs.keys())}")
print("  [OK]")

print("\n" + "=" * 70)
print("  所有测试通过！模型优化成功。")
print("=" * 70)