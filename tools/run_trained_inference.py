"""
使用训练好的模型跑测试集推理。

从测试集 queries.json 读取查询，加载最佳训练 checkpoint 进行推理，
将带预测框的结果图片保存到 test_results/trained/，并生成 summary.json。

用法:
    python tools/run_trained_inference.py --num_samples 50
    python tools/run_trained_inference.py --num_samples 100 --checkpoint output/grounding_dino_multi_modal_epoch_4_loss_2.4251.pth
"""
import os
import sys
import json
import argparse
import importlib.util
import torch
import cv2
import numpy as np
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def import_groundingdino():
    """动态导入 groundingdino 包（兼容直接运行脚本）"""
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


def load_image(img_path, modality, target_size=(512, 512)):
    """加载图像并归一化，返回 (tensor, orig_size)。"""
    if modality == 'depth':
        img = None
        try:
            pil_img = Image.open(img_path)
            img = np.array(pil_img)
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        except Exception:
            pass
        if img is None:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None, None
        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
        if img.dtype == np.uint16 or img.dtype == np.int32:
            img = (img / 65535.0).astype(np.float32)
        else:
            img = img.astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        else:
            img = img.transpose(2, 0, 1)
        return torch.from_numpy(img), (orig_w, orig_h)

    pil_img = Image.open(img_path).convert('RGB' if modality == 'rgb' else 'L')
    orig_size = pil_img.size
    pil_img = pil_img.resize(target_size, Image.BILINEAR)
    arr = np.array(pil_img)
    if modality == 'ir':
        arr = np.expand_dims(arr, axis=0).astype(np.float32) / 255.0
    else:
        arr = arr.transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(arr), orig_size


def denormalize_box(box, orig_size):
    """cxcywh(归一化) -> xyxy(像素)"""
    orig_w, orig_h = orig_size
    cx, cy, w, h = box
    x1 = (cx - w / 2) * orig_w
    y1 = (cy - h / 2) * orig_h
    x2 = (cx + w / 2) * orig_w
    y2 = (cy + h / 2) * orig_h
    return [x1, y1, x2, y2]


def draw_result(rgb_img, pred_boxes, pred_scores, query_text, orig_size,
                score_threshold=0.3, topk=5):
    """在 RGB 图上画预测框 + 文本标签，返回带标注的图。"""
    h, w = rgb_img.shape[:2]
    result = rgb_img.copy()

    # 按分数排序，取 top-k
    sorted_idx = np.argsort(pred_scores)[::-1][:topk]

    colors = [(0, 255, 0), (0, 200, 255), (255, 200, 0), (255, 0, 200), (200, 0, 255)]

    drawn = 0
    for rank, idx in enumerate(sorted_idx):
        score = pred_scores[idx]
        if score < score_threshold and drawn > 0:
            break

        box = pred_boxes[idx]
        x1, y1, x2, y2 = denormalize_box(box, orig_size)
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))

        color = colors[rank % len(colors)]
        thickness = 3 if rank == 0 else 2
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        label = f'#{rank+1} {score:.3f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(result, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(result, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        drawn += 1

    # 顶部写查询文本
    text_line = query_text[:100] + ('...' if len(query_text) > 100 else '')
    cv2.rectangle(result, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.putText(result, text_line, (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return result, drawn


def find_best_checkpoint(output_dir):
    """自动查找最佳 checkpoint（val_loss 最低的）。"""
    checkpoints = []
    for f in os.listdir(output_dir):
        if f.startswith('grounding_dino_multi_modal_epoch_') and f.endswith('.pth'):
            parts = f.replace('.pth', '').split('_loss_')
            if len(parts) == 2:
                try:
                    epoch = int(parts[0].split('_')[-1])
                    val_loss = float(parts[1])
                    checkpoints.append((val_loss, epoch, os.path.join(output_dir, f)))
                except ValueError:
                    continue
    if not checkpoints:
        return None
    checkpoints.sort()
    return checkpoints[0][2]  # 返回 val_loss 最低的


def main():
    parser = argparse.ArgumentParser(description='训练模型推理测试集')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='跑多少条查询 (默认 50)')
    parser.add_argument('--output', type=str, default='./test_results/trained',
                        help='输出目录')
    parser.add_argument('--test_data', type=str,
                        default='./初赛数据集-基于大模型的多模态视觉理解与推理',
                        help='测试集根目录')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint 路径 (不指定则自动查找最佳)')
    parser.add_argument('--score_threshold', type=float, default=0.3,
                        help='分数阈值 (低于此值不画框)')
    parser.add_argument('--topk', type=int, default=5,
                        help='每张图最多画多少个框')
    args = parser.parse_args()

    test_data_root = os.path.join(project_root, args.test_data)
    queries_file = os.path.join(test_data_root, 'queries', 'queries.json')
    output_dir = os.path.join(project_root, args.output)
    os.makedirs(output_dir, exist_ok=True)

    # 加载 queries
    with open(queries_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    query_ids = list(queries.keys())
    total_queries = len(query_ids)
    num_samples = min(args.num_samples, total_queries)
    print(f"测试集查询总数: {total_queries}, 本次推理: {num_samples}")

    # 加载模型
    get_cfg, GroundingDINO = import_groundingdino()
    cfg = get_cfg()

    # 加载训练配置文件（确保 HIDDEN_DIM 等参数与训练时一致）
    config_path = os.path.join(project_root, 'groundingdino', 'config', 'GroundingDINO_Fused_Train.py')
    if os.path.exists(config_path):
        exec(open(config_path).read())
        cfg = get_cfg()
        print(f"Loaded config from: {config_path}")
    else:
        print(f"⚠️ 训练配置未找到: {config_path}，使用默认配置（可能与训练不一致）")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"HIDDEN_DIM={cfg.MODEL.HIDDEN_DIM}, NECK.OUT_CHANNEL={cfg.MODEL.NECK.OUT_CHANNEL}")

    model = GroundingDINO(cfg).to(device)

    # 查找并加载 checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = find_best_checkpoint(os.path.join(project_root, 'output'))
    if ckpt_path is not None:
        ckpt_path = os.path.join(project_root, ckpt_path) if not os.path.isabs(ckpt_path) else ckpt_path
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint 不存在: {ckpt_path}")
            return
        print(f"加载训练 checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '?')
        best_loss = checkpoint.get('best_loss', '?')
        print(f"  Epoch: {epoch}, Best Loss: {best_loss}")
    else:
        print("⚠️ 未找到 checkpoint，使用随机初始化权重！")

    model.eval()

    # Tokenizer
    try:
        from transformers import BertTokenizerFast
        try:
            tokenizer = BertTokenizerFast.from_pretrained(
                cfg.MODEL.TEXT_ENCODER.NAME, local_files_only=True)
            print(f"Loaded BertTokenizerFast from local cache")
        except Exception:
            tokenizer = BertTokenizerFast.from_pretrained(
                cfg.MODEL.TEXT_ENCODER.NAME)
            print(f"Loaded BertTokenizerFast")
    except Exception:
        from groundingdino.datasets.dataset import SimpleTokenizer
        tokenizer = SimpleTokenizer()
        print("Using SimpleTokenizer fallback")

    results = []
    scores_all = []
    boxes_drawn_all = []

    print(f"\n{'=' * 60}")
    print(f"开始推理 (共 {num_samples} 条)")
    print(f"{'=' * 60}")

    for idx in range(num_samples):
        qid = query_ids[idx]
        item = queries[qid]
        query_text = item['query']

        inputs = {}
        orig_size = None

        modality_map = {'visible': 'rgb', 'infrared': 'ir', 'depth': 'depth'}
        for q_key, m_key in modality_map.items():
            if q_key in item and item[q_key]:
                img_path = os.path.join(test_data_root, item[q_key])
                if os.path.exists(img_path):
                    img, size = load_image(img_path, m_key)
                    if img is not None:
                        inputs[m_key] = img.unsqueeze(0).to(device)
                        if orig_size is None and m_key == 'rgb':
                            orig_size = size

        if orig_size is None:
            print(f"  [{idx + 1}/{num_samples}] {qid}: 跳过 (无 RGB 图像)")
            continue

        # 文本编码
        encoding = tokenizer(query_text, padding='max_length', truncation=True,
                             max_length=256, return_tensors='pt')
        inputs['text_input_ids'] = encoding['input_ids'].to(device)
        inputs['text_attention_mask'] = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(inputs)

        pred_logits = outputs['pred_logits']  # [1, N, C]
        pred_boxes = outputs['pred_boxes']    # [1, N, 4]

        # 计算分数
        scores = torch.sigmoid(pred_logits).squeeze(-1)  # [1, N]
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        max_scores, _ = scores.max(dim=-1)  # [1]
        best_idx = torch.argmax(max_scores[0])
        best_score = float(max_scores[0][best_idx].cpu().numpy())
        best_box = pred_boxes[0][best_idx].cpu().numpy()
        scores_all.append(best_score)

        # 所有框的分数和坐标（用于画 top-k）
        all_scores = max_scores[0].cpu().numpy()
        all_boxes = pred_boxes[0].cpu().numpy()

        # 读取原始 RGB 画框
        rgb_path = os.path.join(test_data_root, item['visible'])
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            pil = Image.open(rgb_path).convert('RGB')
            rgb_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        result_img, drawn = draw_result(
            rgb_img, all_boxes, all_scores, query_text, orig_size,
            score_threshold=args.score_threshold, topk=args.topk
        )
        boxes_drawn_all.append(drawn)

        save_path = os.path.join(output_dir, f'{qid}.jpg')
        cv2.imwrite(save_path, result_img)

        rel_save = os.path.relpath(save_path, project_root).replace('\\', '/')
        results.append({
            'query_id': qid,
            'query': query_text,
            'best_score': best_score,
            'best_box_norm': best_box.tolist(),
            'boxes_drawn': drawn,
            'model_type': 'trained',
            'save_path': rel_save,
        })

        print(f"  [{idx + 1}/{num_samples}] {qid}: score={best_score:.4f} "
              f"box=[{best_box[0]:.3f},{best_box[1]:.3f},{best_box[2]:.3f},{best_box[3]:.3f}] "
              f"drawn={drawn}")

    # 汇总
    avg_score = sum(scores_all) / len(scores_all) if scores_all else 0
    avg_drawn = sum(boxes_drawn_all) / len(boxes_drawn_all) if boxes_drawn_all else 0
    summary = {
        'model_type': 'trained',
        'checkpoint': os.path.basename(ckpt_path) if ckpt_path else 'random_init',
        'num_queries': len(results),
        'total_queries_in_file': total_queries,
        'avg_score': avg_score,
        'max_score': max(scores_all) if scores_all else 0,
        'min_score': min(scores_all) if scores_all else 0,
        'avg_boxes_drawn': avg_drawn,
        'score_threshold': args.score_threshold,
        'above_threshold': sum(1 for s in scores_all if s >= args.score_threshold),
        'results': results,
    }
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"推理完成！")
    print(f"Checkpoint: {os.path.basename(ckpt_path) if ckpt_path else 'random_init'}")
    print(f"结果目录: {output_dir}")
    print(f"图片数量: {len(results)}")
    print(f"汇总文件: {summary_path}")
    print(f"平均 score: {avg_score:.4f}")
    print(f"最高 score: {max(scores_all) if scores_all else 0:.4f}")
    print(f"最低 score: {min(scores_all) if scores_all else 0:.4f}")
    print(f"平均画框数: {avg_drawn:.1f}")
    print(f"超过阈值({args.score_threshold})的数量: {summary['above_threshold']}/{len(results)}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
