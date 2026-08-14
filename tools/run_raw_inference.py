"""
裸跑测试集推理脚本（Raw Model，无微调权重）

从测试集 queries.json 读取查询，用随机初始化的 GroundingDINO 模型推理，
将带预测框的结果图片保存到 test_results/raw/，并生成 summary.json。

用法:
    python tools/run_raw_inference.py --num_samples 50
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

    import_groundingdino_root = _load('groundingdino', [])
    _load('groundingdino.config', ['config'])
    _load('groundingdino.models', ['models'])
    _load('groundingdino.datasets', ['datasets'])

    from groundingdino.config import get_cfg
    from groundingdino.models import GroundingDINO
    return get_cfg, GroundingDINO


def load_image(img_path, modality, target_size=(512, 512)):
    """加载图像并归一化，返回 (tensor, orig_size)。
    修复：depth 是 16 位 PNG，cv2 读不了，用 PIL 读取。
    """
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


def draw_result(rgb_img, pred_box_norm, pred_score, query_text, orig_size):
    """在 RGB 图上画预测框 + 文本标签，返回带标注的图。"""
    h, w = rgb_img.shape[:2]
    result = rgb_img.copy()
    x1, y1, x2, y2 = denormalize_box(pred_box_norm, orig_size)
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))

    color = (0, 255, 0)
    cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)

    label = f'Pred: {float(pred_score):.3f}'
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(result, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
    cv2.putText(result, label, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # 顶部写查询文本
    text_line = query_text[:100] + ('...' if len(query_text) > 100 else '')
    cv2.rectangle(result, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.putText(result, text_line, (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return result


def main():
    parser = argparse.ArgumentParser(description='裸跑测试集推理 (Raw Model)')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='跑多少条查询 (默认 50)')
    parser.add_argument('--output', type=str, default='./test_results/raw',
                        help='输出目录')
    parser.add_argument('--test_data', type=str,
                        default='./初赛数据集-基于大模型的多模态视觉理解与推理',
                        help='测试集根目录')
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
    print(f"测试集查询总数: {total_queries}, 本次裸跑: {num_samples}")

    # 加载模型 (raw, 随机初始化)
    get_cfg, GroundingDINO = import_groundingdino()
    cfg = get_cfg()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print("加载 Raw 模型 (随机初始化, 无微调权重)...")
    model = GroundingDINO(cfg).to(device)
    model.eval()

    # Tokenizer (离线优先)
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

    print(f"\n{'=' * 60}")
    print(f"开始裸跑推理 (共 {num_samples} 条)")
    print(f"{'=' * 60}")

    for idx in range(num_samples):
        qid = query_ids[idx]
        item = queries[qid]
        query_text = item['query']

        inputs = {}
        orig_size = None

        # 模态映射: queries.json 用 visible/infrared/depth, 模型用 rgb/ir/depth
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
        scores = torch.sigmoid(pred_logits).squeeze(-1)  # [1, N]
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        # 每个查询框的最高分数
        max_scores, _ = scores.max(dim=-1)  # [1]
        best_idx = torch.argmax(max_scores[0])
        pred_box = pred_boxes[0][best_idx].cpu().numpy()
        pred_score = float(max_scores[0][best_idx].cpu().numpy())
        scores_all.append(pred_score)

        # 读取原始 RGB 画框
        rgb_path = os.path.join(test_data_root, item['visible'])
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            pil = Image.open(rgb_path).convert('RGB')
            rgb_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        result_img = draw_result(rgb_img, pred_box, pred_score, query_text, orig_size)
        save_path = os.path.join(output_dir, f'{qid}.jpg')
        cv2.imwrite(save_path, result_img)

        rel_save = os.path.relpath(save_path, project_root).replace('\\', '/')
        results.append({
            'query_id': qid,
            'query': query_text,
            'score': pred_score,
            'pred_box_norm': pred_box.tolist(),
            'model_type': 'raw',
            'save_path': rel_save,
        })

        print(f"  [{idx + 1}/{num_samples}] {qid}: score={pred_score:.4f} "
              f"box=[{pred_box[0]:.3f},{pred_box[1]:.3f},{pred_box[2]:.3f},{pred_box[3]:.3f}]")

    # 汇总
    avg_score = sum(scores_all) / len(scores_all) if scores_all else 0
    summary = {
        'model_type': 'raw',
        'num_queries': len(results),
        'total_queries_in_file': total_queries,
        'avg_score': avg_score,
        'max_score': max(scores_all) if scores_all else 0,
        'min_score': min(scores_all) if scores_all else 0,
        'threshold': 0.3,
        'above_threshold': sum(1 for s in scores_all if s >= 0.3),
        'results': results,
    }
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"裸跑推理完成！")
    print(f"结果目录: {output_dir}")
    print(f"图片数量: {len(results)}")
    print(f"汇总文件: {summary_path}")
    print(f"平均 score: {avg_score:.4f}")
    print(f"最高 score: {max(scores_all) if scores_all else 0:.4f}")
    print(f"最低 score: {min(scores_all) if scores_all else 0:.4f}")
    print(f"超过阈值(0.3)的数量: {summary['above_threshold']}/{len(results)}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
