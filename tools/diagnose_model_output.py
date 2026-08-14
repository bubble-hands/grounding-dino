"""
诊断模型输出坐标分布问题。
检查模型在训练数据上的实际预测值，以及反归一化是否正确。
"""
import os
import sys
import json
import importlib.util
import torch
import numpy as np
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def import_groundingdino():
    """动态导入 groundingdino 包"""
    gdino_path = os.path.join(project_root, 'groundingdino')

    def _load(mod_name, sub_path):
        init_path = os.path.join(gdino_path, *sub_path, '__init__.py')
        spec = importlib.util.spec_from_file_location(mod_name, init_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return module

    _load('groundingdino', [])
    _load('groundingdino.config', ['config'])
    _load('groundingdino.models', ['models'])
    _load('groundingdino.datasets', ['datasets'])

    from groundingdino.config import get_cfg
    from groundingdino.models import GroundingDINO
    return get_cfg, GroundingDINO


def load_image(img_path, target_size=(512, 512)):
    """加载图像，返回 (tensor, orig_size)"""
    pil_img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = pil_img.size
    pil_img = pil_img.resize(target_size, Image.BILINEAR)
    arr = np.array(pil_img).transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(arr), (orig_w, orig_h)


def main():
    # 加载训练数据
    train_json = os.path.join(project_root, 'data', 'train.json')
    with open(train_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"训练样本数: {len(data)}")

    # 加载模型
    get_cfg, GroundingDINO = import_groundingdino()
    cfg = get_cfg()

    # 加载训练配置
    config_path = os.path.join(project_root, 'groundingdino', 'config', 'GroundingDINO_Fused_Train.py')
    if os.path.exists(config_path):
        exec(open(config_path).read())
        cfg = get_cfg()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    model = GroundingDINO(cfg).to(device)

    # 查找 checkpoint
    output_dir = os.path.join(project_root, 'output')
    best_ckpt = None
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.startswith('grounding_dino_multi_modal_epoch_') and f.endswith('.pth'):
                if best_ckpt is None or f < best_ckpt:
                    best_ckpt = os.path.join(output_dir, f)

    if best_ckpt and os.path.exists(best_ckpt):
        print(f"加载 checkpoint: {best_ckpt}")
        checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("⚠️ 未找到 checkpoint，使用随机初始化权重")

    model.eval()

    # Tokenizer
    try:
        from transformers import BertTokenizerFast
        try:
            tokenizer = BertTokenizerFast.from_pretrained(
                cfg.MODEL.TEXT_ENCODER.NAME, local_files_only=True)
        except Exception:
            tokenizer = BertTokenizerFast.from_pretrained(
                cfg.MODEL.TEXT_ENCODER.NAME)
    except Exception:
        from groundingdino.datasets.dataset import SimpleTokenizer
        tokenizer = SimpleTokenizer()

    # 测试前几个样本
    print("\n" + "="*60)
    print("诊断模型输出坐标")
    print("="*60)

    all_pred_boxes = []
    all_gt_boxes = []

    for idx in range(min(10, len(data))):
        item = data[idx]
        rgb_path = item['rgb']

        if not os.path.exists(rgb_path):
            print(f"样本 {idx}: 图像不存在，跳过")
            continue

        # 加载图像
        img, orig_size = load_image(rgb_path)
        orig_w, orig_h = orig_size

        # 准备输入
        inputs = {
            'rgb': img.unsqueeze(0).to(device),
        }

        # 文本编码
        text = item.get('text', 'target')
        encoding = tokenizer(text, padding='max_length', truncation=True,
                             max_length=256, return_tensors='pt')
        inputs['text_input_ids'] = encoding['input_ids'].to(device)
        inputs['text_attention_mask'] = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(inputs)

        pred_boxes = outputs['pred_boxes'][0].cpu().numpy()  # [N, 4] cxcywh 归一化
        pred_logits = outputs['pred_logits'][0].cpu().numpy()  # [N, C]

        # 获取 GT 框
        ann = item['annotations'][0]
        gt_x, gt_y, gt_w, gt_h = ann['bbox']
        gt_cx = (gt_x + gt_w / 2) / orig_w
        gt_cy = (gt_y + gt_h / 2) / orig_h
        gt_nw = gt_w / orig_w
        gt_nh = gt_h / orig_h

        # 统计所有预测框的坐标分布
        all_pred_boxes.append(pred_boxes)
        all_gt_boxes.append([gt_cx, gt_cy, gt_nw, gt_nh])

        # 找最高分的预测框
        scores = torch.sigmoid(torch.from_numpy(pred_logits)).squeeze(-1)
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)
        max_scores, _ = scores.max(dim=-1)
        best_idx = torch.argmax(max_scores[0])
        best_box = pred_boxes[best_idx]

        # 反归一化到像素坐标
        pred_cx, pred_cy, pred_w, pred_h = best_box
        pred_x1 = (pred_cx - pred_w / 2) * orig_w
        pred_y1 = (pred_cy - pred_h / 2) * orig_h
        pred_x2 = (pred_cx + pred_w / 2) * orig_w
        pred_y2 = (pred_cy + pred_h / 2) * orig_h

        gt_x1 = gt_x
        gt_y1 = gt_y
        gt_x2 = gt_x + gt_w
        gt_y2 = gt_y + gt_h

        print(f"\n样本 {idx}: {item['query_id']}")
        print(f"  GT 框 (归一化): [{gt_cx:.4f}, {gt_cy:.4f}, {gt_nw:.4f}, {gt_nh:.4f}]")
        print(f"  GT 框 (像素):   [{gt_x1:.1f}, {gt_y1:.1f}, {gt_x2:.1f}, {gt_y2:.1f}]")
        print(f"  Pred 框 (归一化): [{pred_cx:.4f}, {pred_cy:.4f}, {pred_w:.4f}, {pred_h:.4f}]")
        print(f"  Pred 框 (像素):   [{pred_x1:.1f}, {pred_y1:.1f}, {pred_x2:.1f}, {pred_y2:.1f}]")
        print(f"  图像尺寸: {orig_w}x{orig_h}")
        print(f"  预测框数量: {len(pred_boxes)}")

    # 汇总统计
    if all_pred_boxes:
        all_pred_boxes = np.concatenate(all_pred_boxes, axis=0)

        print("\n" + "="*60)
        print("所有预测框坐标分布统计 (归一化)")
        print("="*60)
        print(f"cx: min={all_pred_boxes[:,0].min():.4f}, max={all_pred_boxes[:,0].max():.4f}, mean={all_pred_boxes[:,0].mean():.4f}")
        print(f"cy: min={all_pred_boxes[:,1].min():.4f}, max={all_pred_boxes[:,1].max():.4f}, mean={all_pred_boxes[:,1].mean():.4f}")
        print(f"w: min={all_pred_boxes[:,2].min():.4f}, max={all_pred_boxes[:,2].max():.4f}, mean={all_pred_boxes[:,2].mean():.4f}")
        print(f"h: min={all_pred_boxes[:,3].min():.4f}, max={all_pred_boxes[:,3].max():.4f}, mean={all_pred_boxes[:,3].mean():.4f}")

        # 象限分布
        cx = all_pred_boxes[:, 0]
        cy = all_pred_boxes[:, 1]
        left = cx < 0.5
        right = cx >= 0.5
        top = cy < 0.5
        bottom = cy >= 0.5

        print(f"\n左下象限 (cx<0.5, cy>=0.5): {(left & bottom).sum()} ({(left & bottom).mean()*100:.1f}%)")
        print(f"左上象限 (cx<0.5, cy<0.5): {(left & top).sum()} ({(left & top).mean()*100:.1f}%)")
        print(f"右上象限 (cx>=0.5, cy<0.5): {(right & top).sum()} ({(right & top).mean()*100:.1f}%)")
        print(f"右下象限 (cx>=0.5, cy>=0.5): {(right & bottom).sum()} ({(right & bottom).mean()*100:.1f}%)")

        # GT 框统计
        all_gt_boxes = np.array(all_gt_boxes)
        print("\n" + "="*60)
        print("GT 框坐标分布统计 (归一化)")
        print("="*60)
        print(f"cx: min={all_gt_boxes[:,0].min():.4f}, max={all_gt_boxes[:,0].max():.4f}, mean={all_gt_boxes[:,0].mean():.4f}")
        print(f"cy: min={all_gt_boxes[:,1].min():.4f}, max={all_gt_boxes[:,1].max():.4f}, mean={all_gt_boxes[:,1].mean():.4f}")


if __name__ == '__main__':
    main()