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
    if modality == 'depth':
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None, None
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
        if img.dtype == np.uint16:
            img = (img / 65535.0).astype(np.float32)
        else:
            img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
    else:
        img = Image.open(img_path).convert('RGB' if modality == 'rgb' else 'L')
        if img is None:
            return None, None
        orig_size = img.size
        img = img.resize(target_size, Image.BILINEAR)
        img = np.array(img)
        if modality == 'ir':
            img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        else:
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(img), orig_size if modality != 'depth' else target_size


def denormalize_bbox(box, orig_size):
    orig_w, orig_h = orig_size
    cx, cy, w, h = box
    x1 = (cx - w/2) * orig_w
    y1 = (cy - h/2) * orig_h
    x2 = (cx + w/2) * orig_w
    y2 = (cy + h/2) * orig_h
    return [x1, y1, x2, y2]


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def draw_single_pred_box(img, pred_box, pred_score, orig_size):
    h, w = img.shape[:2]
    result = img.copy()
    
    box_list = pred_box.tolist() if hasattr(pred_box, 'tolist') else pred_box
    x1, y1, x2, y2 = denormalize_bbox(box_list, orig_size)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    score_val = float(pred_score) if hasattr(pred_score, 'item') else pred_score
    
    color = (0, 255, 0)
    thickness = 3
    cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
    
    label = f'Pred: {score_val:.2f}'
    font_scale = 0.7
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    cv2.rectangle(result, (x1, y1-th-8), (x1+tw+8, y1), color, -1)
    cv2.putText(result, label, (x1+4, y1-6),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
    
    return result


def draw_gt_box(img, annotations):
    result = img.copy()
    for ann in annotations:
        bbox = ann['bbox']
        x, y, w, h = bbox
        x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
        color = (0, 0, 255)
        thickness = 3
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        label = 'GT'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(result, (x1, y1-th-8), (x1+tw+8, y1), color, -1)
        cv2.putText(result, label, (x1+4, y1-6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return result


def draw_comparison(rgb_img, pred_box, pred_score, annotations, orig_size, text_label, iou_score):
    h, w = rgb_img.shape[:2]
    gap = 10
    panel_width = w
    panel_height = h
    
    title_height = 50
    info_height = 80
    canvas_width = panel_width * 3 + gap * 2
    canvas_height = panel_height + title_height + info_height
    
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 240
    
    cv2.putText(canvas, 'Original Image', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    cv2.putText(canvas, 'Prediction (Green)', (panel_width + gap + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)
    cv2.putText(canvas, 'Ground Truth (Red)', (2*panel_width + 2*gap + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
    
    canvas[title_height:title_height+panel_height, 0:panel_width] = rgb_img
    
    pred_img = draw_single_pred_box(rgb_img, pred_box, pred_score, orig_size)
    canvas[title_height:title_height+panel_height, panel_width+gap:2*panel_width+gap] = pred_img
    
    gt_img = draw_gt_box(rgb_img, annotations)
    canvas[title_height:title_height+panel_height, 2*panel_width+2*gap:3*panel_width+2*gap] = gt_img
    
    info_y = title_height + panel_height + 30
    cv2.putText(canvas, f'Text: {text_label[:90]}', (10, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    if len(text_label) > 90:
        cv2.putText(canvas, text_label[90:], (10, info_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    score_text = f'Score: {float(pred_score):.3f}  |  IoU: {iou_score:.3f}'
    cv2.putText(canvas, score_text, (canvas_width - 350, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 200), 2)
    
    return canvas


def main():
    parser = argparse.ArgumentParser(description='Visualize Validation Results with Single Box')
    parser.add_argument('--config', type=str, 
                        default='groundingdino/config/GroundingDINO_SwinT_MultiModal.py')
    parser.add_argument('--checkpoint', type=str, 
                        default='output/grounding_dino_multi_modal_epoch_4_loss_11.4426.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='./data', help='Data directory')
    parser.add_argument('--output', type=str, default='./vis_results', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--threshold', type=float, default=0.5, help='Score threshold for display')
    args = parser.parse_args()

    cfg = get_cfg()
    
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    if os.path.exists(config_path):
        exec(open(config_path).read())
        cfg = get_cfg()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(project_root, checkpoint_path)
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints in output/:")
        output_dir = os.path.join(project_root, 'output')
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                if f.endswith('.pth'):
                    print(f"  - output/{f}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = GroundingDINO(cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    epoch = checkpoint.get('epoch', '?')
    val_loss = checkpoint.get('best_loss', '?')
    print(f"Loaded model from epoch {epoch}, val loss: {val_loss}")

    ann_file = os.path.join(project_root, args.data, 'val.json')
    with open(ann_file, 'r') as f:
        val_data = json.load(f)

    os.makedirs(args.output, exist_ok=True)
    num_samples = min(args.num_samples, len(val_data))
    
    from groundingdino.datasets.dataset import SimpleTokenizer
    tokenizer = SimpleTokenizer()
    results = []
    total_iou = 0.0

    for idx in range(num_samples):
        item = val_data[idx]
        print(f"\nProcessing sample {idx+1}/{num_samples}...")

        inputs = {}
        orig_size = None
        
        for modality in cfg.MODEL.MULTI_MODAL.MODALITIES:
            if modality in item and item[modality] is not None:
                img_path = os.path.join(project_root, args.data, item[modality])
                if os.path.exists(img_path):
                    img, size = load_image(img_path, modality)
                    if img is not None:
                        inputs[modality] = img.unsqueeze(0).to(device)
                        if orig_size is None:
                            orig_size = size

        text = item.get('text', '')
        encoding = tokenizer(text, padding='max_length', truncation=True, 
                           max_length=256, return_tensors='pt')
        inputs['text_input_ids'] = encoding['input_ids'].to(device)
        inputs['text_attention_mask'] = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(inputs)

        if 'results' in outputs:
            detections = outputs['results'][0]
            pred_box = detections['boxes'][0].cpu().numpy() if torch.is_tensor(detections['boxes']) else detections['boxes'][0]
            pred_score = detections['scores'][0].cpu().numpy() if torch.is_tensor(detections['scores']) else detections['scores'][0]
        else:
            pred_logits = outputs['pred_logits']
            pred_boxes = outputs['pred_boxes']
            scores = torch.sigmoid(pred_logits).squeeze(-1)
            max_scores, _ = scores.max(dim=-1)
            best_idx = torch.argmax(max_scores[0])
            pred_box = pred_boxes[0][best_idx].cpu().numpy()
            pred_score = max_scores[0][best_idx].cpu().numpy()

        rgb_path = os.path.join(project_root, args.data, item.get('rgb', ''))
        if os.path.exists(rgb_path) and orig_size is not None:
            rgb_img = cv2.imread(rgb_path)
            if rgb_img is None:
                print(f"  Warning: Could not read {rgb_path}")
                continue
            
            pred_box_orig = denormalize_bbox(pred_box, orig_size)
            
            annotations = item.get('annotations', [])
            gt_box_orig = None
            if annotations:
                gt_bbox = annotations[0]['bbox']
                gt_box_orig = [gt_bbox[0], gt_bbox[1], gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]]
            
            iou_score = 0.0
            if gt_box_orig is not None:
                iou_score = compute_iou(pred_box_orig, gt_box_orig)
            
            total_iou += iou_score
            
            comparison_img = draw_comparison(rgb_img, pred_box, pred_score, annotations, orig_size, text, iou_score)
            save_path = os.path.join(args.output, f'sample_{idx:03d}.jpg')
            cv2.imwrite(save_path, comparison_img)
            
            print(f"  Saved: {save_path}")
            print(f"  Text: {text}")
            print(f"  Score: {float(pred_score):.3f}")
            print(f"  IoU with GT: {iou_score:.3f}")
            print(f"  Pred box (normalized): [{pred_box[0]:.3f}, {pred_box[1]:.3f}, {pred_box[2]:.3f}, {pred_box[3]:.3f}]")
            print(f"  Pred box (pixels): [{pred_box_orig[0]:.0f}, {pred_box_orig[1]:.0f}, {pred_box_orig[2]:.0f}, {pred_box_orig[3]:.0f}]")
            if gt_box_orig:
                print(f"  GT box (pixels): [{gt_box_orig[0]:.0f}, {gt_box_orig[1]:.0f}, {gt_box_orig[2]:.0f}, {gt_box_orig[3]:.0f}]")
            
            results.append({
                'idx': idx,
                'text': text,
                'score': float(pred_score),
                'iou': iou_score,
                'pred_box_norm': pred_box.tolist(),
                'pred_box_pixels': [int(x) for x in pred_box_orig],
                'save_path': save_path
            })

    avg_iou = total_iou / len(results) if results else 0
    summary_path = os.path.join(args.output, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'average_iou': avg_iou,
            'num_samples': len(results),
            'results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Visualization completed!")
    print(f"Results saved to: {args.output}")
    print(f"Summary: {summary_path}")
    print(f"\nAverage IoU: {avg_iou:.3f}")
    
    if results:
        print(f"\nDetailed results:")
        for r in results:
            print(f"  Sample {r['idx']:03d}: Score={r['score']:.3f}, IoU={r['iou']:.3f}")
            print(f"    Text: {r['text'][:60]}...")


if __name__ == '__main__':
    main()
