import os
import sys
import json
import argparse
import importlib.util
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def import_groundingdino():
    groundingdino_path = os.path.join(project_root, 'groundingdino')
    
    init_path = os.path.join(groundingdino_path, '__init__.py')
    spec = importlib.util.spec_from_file_location('groundingdino', init_path)
    groundingdino = importlib.util.module_from_spec(spec)
    sys.modules['groundingdino'] = groundingdino
    spec.loader.exec_module(groundingdino)
    
    config_path = os.path.join(groundingdino_path, 'config', '__init__.py')
    spec_config = importlib.util.spec_from_file_location('groundingdino.config', config_path)
    config_module = importlib.util.module_from_spec(spec_config)
    sys.modules['groundingdino.config'] = config_module
    spec_config.loader.exec_module(config_module)
    
    models_path = os.path.join(groundingdino_path, 'models', '__init__.py')
    spec_models = importlib.util.spec_from_file_location('groundingdino.models', models_path)
    models_module = importlib.util.module_from_spec(spec_models)
    sys.modules['groundingdino.models'] = models_module
    spec_models.loader.exec_module(models_module)
    
    datasets_path = os.path.join(groundingdino_path, 'datasets', '__init__.py')
    spec_datasets = importlib.util.spec_from_file_location('groundingdino.datasets', datasets_path)
    datasets_module = importlib.util.module_from_spec(spec_datasets)
    sys.modules['groundingdino.datasets'] = datasets_module
    spec_datasets.loader.exec_module(datasets_module)
    
    return config_module.get_cfg, models_module.GroundingDINO


get_cfg, GroundingDINO = import_groundingdino()


def load_image(img_path, modality, target_size=(512, 512)):
    if not os.path.exists(img_path):
        return None, None
    
    if modality == 'depth':
        pil_img = Image.open(img_path)
        img = np.array(pil_img)
        if img.ndim == 3:
            img = img[:, :, 0]
        orig_size = (img.shape[1], img.shape[0])
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
        if img.dtype == np.uint16:
            img = (img / 65535.0).astype(np.float32)
        elif img.dtype == np.float32:
            pass
        else:
            img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return torch.from_numpy(img), orig_size
    elif modality == 'infrared':
        img = Image.open(img_path).convert('L')
        if img is None:
            return None, None
        orig_size = img.size
        img = img.resize(target_size, Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return torch.from_numpy(img), orig_size
    else:
        img = Image.open(img_path).convert('RGB')
        if img is None:
            return None, None
        orig_size = img.size
        img = img.resize(target_size, Image.BILINEAR)
        img = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
        return torch.from_numpy(img), orig_size


def denormalize_bbox(box, orig_size):
    orig_w, orig_h = orig_size
    cx, cy, w, h = box
    x1 = (cx - w/2) * orig_w
    y1 = (cy - h/2) * orig_h
    x2 = (cx + w/2) * orig_w
    y2 = (cy + h/2) * orig_h
    return [x1, y1, x2, y2]


def draw_pred_boxes(img, pred_boxes, pred_scores, orig_size, threshold=0.3, max_boxes=5):
    h, w = img.shape[:2]
    result = img.copy()
    
    if len(pred_boxes) == 0:
        return result
    
    scores = np.array(pred_scores)
    sorted_idx = np.argsort(scores)[::-1]
    
    drawn_count = 0
    for idx in sorted_idx:
        if drawn_count >= max_boxes:
            break
        if scores[idx] < threshold:
            break
        
        box = pred_boxes[idx]
        box_list = box.tolist() if hasattr(box, 'tolist') else box
        x1, y1, x2, y2 = denormalize_bbox(box_list, orig_size)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        score_val = float(scores[idx])
        
        color = (0, 255, 0)
        thickness = 3
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        label = f'{score_val:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(result, (x1, y1-th-6), (x1+tw+6, y1), color, -1)
        cv2.putText(result, label, (x1+3, y1-4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        drawn_count += 1
    
    return result


def draw_single_pred_box(img, pred_box, pred_score, orig_size):
    h, w = img.shape[:2]
    result = img.copy()
    
    box_list = pred_box.tolist() if hasattr(pred_box, 'tolist') else pred_box
    x1, y1, x2, y2 = denormalize_bbox(box_list, orig_size)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    score_val = float(pred_score)
    
    color = (0, 255, 0)
    thickness = 4
    cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
    
    label = f'Score: {score_val:.3f}'
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(result, (x1, y1-th-8), (x1+tw+8, y1), color, -1)
    cv2.putText(result, label, (x1+4, y1-5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return result


def create_visualization(img, pred_box, pred_score, orig_size, query_text, query_id, model_type):
    h, w = img.shape[:2]
    gap = 10
    title_height = 80
    info_height = 100
    panel_width = w
    panel_height = h
    canvas_width = panel_width + gap + panel_width
    canvas_height = title_height + panel_height + info_height
    
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 240
    
    color_map = {
        'raw': (100, 100, 255),
        'fine_tuned': (0, 200, 0),
        'default': (0, 255, 0)
    }
    pred_color = color_map.get(model_type, color_map['default'])
    
    title1 = 'Original Image'
    title2 = f'Prediction ({model_type})'
    
    cv2.putText(canvas, title1, (10, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2)
    cv2.putText(canvas, title2, (panel_width + gap + 10, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, pred_color, 2)
    
    canvas[title_height:title_height+panel_height, 0:panel_width] = img
    
    pred_img = draw_single_pred_box(img, pred_box, pred_score, orig_size)
    canvas[title_height:title_height+panel_height, panel_width+gap:2*panel_width+gap] = pred_img
    
    info_y = title_height + panel_height + 25
    cv2.putText(canvas, f'Query ID: {query_id}', (10, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1)
    
    text_display = f'Query: {query_text[:100]}'
    if len(query_text) > 100:
        text_display = query_text[:100]
    cv2.putText(canvas, text_display, (10, info_y + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)
    if len(query_text) > 100:
        cv2.putText(canvas, query_text[100:200], (10, info_y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)
    
    score_text = f'Score: {float(pred_score):.3f}  |  Model: {model_type}'
    cv2.putText(canvas, score_text, (canvas_width - 400, info_y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)
    
    return canvas


def main():
    parser = argparse.ArgumentParser(description='Test Set Inference - Baseline')
    parser.add_argument('--data', type=str, 
                        default='初赛数据集-基于大模型的多模态视觉理解与推理',
                        help='Test set directory')
    parser.add_argument('--queries', type=str, default='queries/queries.json',
                        help='Queries JSON file path (relative to data dir)')
    parser.add_argument('--output', type=str, default='./test_results',
                        help='Output directory')
    parser.add_argument('--model_type', type=str, default='raw',
                        choices=['raw', 'fine_tuned'],
                        help='Model type')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint')
    parser.add_argument('--num_samples', type=int, default=30,
                        help='Number of queries to process')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Score threshold')
    args = parser.parse_args()

    cfg = get_cfg()
    
    data_dir = args.data
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(project_root, data_dir)
    
    queries_file = os.path.join(data_dir, args.queries)
    if not os.path.exists(queries_file):
        print(f"Error: Queries file not found: {queries_file}")
        return

    with open(queries_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    query_ids = list(queries.keys())
    print(f"Total queries in file: {len(query_ids)}")
    
    num_samples = min(args.num_samples, len(query_ids))
    selected_ids = query_ids[:num_samples]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model type: {args.model_type}")

    model = GroundingDINO(cfg).to(device)
    
    if args.model_type == 'fine_tuned' and args.checkpoint:
        checkpoint_path = args.checkpoint
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(project_root, checkpoint_path)
        
        if os.path.exists(checkpoint_path):
            print(f"Loading fine-tuned weights: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            args.model_type = 'raw'
    else:
        print("Using RAW model (random initialization)")

    model.eval()
    
    from groundingdino.datasets.dataset import SimpleTokenizer
    tokenizer = SimpleTokenizer()

    output_dir = os.path.join(args.output, args.model_type)
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for query_id in tqdm(selected_ids, desc="Processing queries"):
        query_data = queries[query_id]
        
        visible_path = os.path.normpath(os.path.join(data_dir, query_data.get('visible', '')))
        infrared_path = os.path.normpath(os.path.join(data_dir, query_data.get('infrared', '')))
        depth_path = os.path.normpath(os.path.join(data_dir, query_data.get('depth', '')))
        query_text = query_data.get('query', '')

        inputs = {}
        orig_size = None
        
        if os.path.exists(visible_path):
            img_rgb, size = load_image(visible_path, 'rgb')
            if img_rgb is not None:
                inputs['rgb'] = img_rgb.unsqueeze(0).to(device)
                if orig_size is None:
                    orig_size = size
        
        if os.path.exists(infrared_path):
            img_ir, _ = load_image(infrared_path, 'infrared')
            if img_ir is not None:
                inputs['ir'] = img_ir.unsqueeze(0).to(device)
        
        if os.path.exists(depth_path):
            img_depth, _ = load_image(depth_path, 'depth')
            if img_depth is not None:
                inputs['depth'] = img_depth.unsqueeze(0).to(device)

        if 'rgb' not in inputs:
            print(f"Warning: No RGB image for query {query_id}")
            continue

        encoding = tokenizer(query_text, padding='max_length', truncation=True,
                           max_length=256, return_tensors='pt')
        inputs['text_input_ids'] = encoding['input_ids'].to(device)
        inputs['text_attention_mask'] = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(inputs)

        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        scores = torch.sigmoid(pred_logits).squeeze(-1)
        max_scores, _ = scores.max(dim=-1)
        best_idx = torch.argmax(max_scores[0])
        pred_box = pred_boxes[0][best_idx].cpu().numpy()
        pred_score = max_scores[0][best_idx].cpu().item()

        if os.path.exists(visible_path) and orig_size is not None:
            pil_img = Image.open(visible_path).convert('RGB')
            rgb_img = np.array(pil_img)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            
            if orig_size != (rgb_img.shape[1], rgb_img.shape[0]):
                rgb_img = cv2.resize(rgb_img, orig_size)
            
            comparison_img = create_visualization(
                rgb_img, pred_box, pred_score, orig_size,
                query_text, query_id, args.model_type
            )
            
            save_path = os.path.join(output_dir, f'{query_id}.jpg')
            cv2.imwrite(save_path, comparison_img)
            
            results.append({
                'query_id': query_id,
                'query': query_text,
                'score': pred_score,
                'pred_box_norm': pred_box.tolist(),
                'model_type': args.model_type,
                'save_path': save_path
            })

    summary = {
        'model_type': args.model_type,
        'num_queries': len(results),
        'total_queries_in_file': len(query_ids),
        'avg_score': sum(r['score'] for r in results) / len(results) if results else 0,
        'max_score': max(r['score'] for r in results) if results else 0,
        'min_score': min(r['score'] for r in results) if results else 0,
        'threshold': args.threshold,
        'above_threshold': sum(1 for r in results if r['score'] >= args.threshold),
        'results': results
    }
    
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Inference completed!")
    print(f"Model: {args.model_type}")
    print(f"Queries processed: {len(results)}")
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"\nScore Statistics:")
    print(f"  Average: {summary['avg_score']:.3f}")
    print(f"  Max: {summary['max_score']:.3f}")
    print(f"  Min: {summary['min_score']:.3f}")
    print(f"  Above threshold ({args.threshold}): {summary['above_threshold']}/{len(results)}")


if __name__ == '__main__':
    main()
