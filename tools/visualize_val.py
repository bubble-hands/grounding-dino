import os
import sys
import json
import argparse
import importlib.util
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
    
    return config_module.get_cfg, models_module.GroundingDINO


get_cfg, GroundingDINO = import_groundingdino()


def load_image(img_path, modality, target_size=(512, 512)):
    if modality == 'depth':
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
        if img.dtype == np.uint16:
            img = (img / 65535.0).astype(np.float32)
        else:
            img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
    else:
        img = Image.open(img_path).convert('RGB' if modality == 'rgb' else 'L')
        orig_size = img.size
        img = img.resize(target_size, Image.BILINEAR)
        img = np.array(img)
        if modality == 'ir':
            img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        else:
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(img), orig_size if modality != 'depth' else target_size


def denormalize_bbox(box, orig_size, target_size=(512, 512)):
    orig_w, orig_h = orig_size
    cx, cy, w, h = box
    x1 = (cx - w/2) * orig_w
    y1 = (cy - h/2) * orig_h
    x2 = (cx + w/2) * orig_w
    y2 = (cy + h/2) * orig_h
    return [x1, y1, x2, y2]


def draw_boxes(img, pred_boxes, pred_scores, orig_size, threshold=0.3):
    h, w = img.shape[:2]
    result = img.copy()
    
    for box, score in zip(pred_boxes, pred_scores):
        if score > threshold:
            box = box.tolist() if hasattr(box, 'tolist') else box
            score = float(score) if hasattr(score, 'item') else score
            x1, y1, x2, y2 = denormalize_bbox(box, orig_size)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result, f'{score:.2f}', (x1, max(15, y1-5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Visualize Validation Results')
    parser.add_argument('--config', type=str, default='groundingdino/config/GroundingDINO_SwinT_MultiModal.py')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--output', type=str, default='./vis_results')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.3)
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

    model = GroundingDINO(cfg).to(device)
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {args.checkpoint}")

    model.eval()

    ann_file = os.path.join(args.data, 'val.json')
    with open(ann_file, 'r') as f:
        val_data = json.load(f)

    os.makedirs(args.output, exist_ok=True)

    num_samples = min(args.num_samples, len(val_data))
    results = []

    dataset_path = os.path.join(project_root, 'groundingdino', 'datasets', 'dataset.py')
    spec_ds = importlib.util.spec_from_file_location('groundingdino.datasets.dataset', dataset_path)
    ds_module = importlib.util.module_from_spec(spec_ds)
    spec_ds.loader.exec_module(ds_module)
    tokenizer = ds_module.SimpleTokenizer()

    for idx in range(num_samples):
        item = val_data[idx]
        print(f"Processing sample {idx+1}/{num_samples}...")

        inputs = {}
        orig_size = None
        
        for modality in cfg.MODEL.MULTI_MODAL.MODALITIES:
            if modality in item and item[modality] is not None:
                img_path = os.path.join(args.data, item[modality])
                if os.path.exists(img_path):
                    img, size = load_image(img_path, modality)
                    inputs[modality] = img.unsqueeze(0).to(device)
                    if orig_size is None:
                        orig_size = size

        text = item.get('text', '')
        encoding = tokenizer(text, padding='max_length', truncation=True, 
                           max_length=256, return_tensors='pt')
        inputs['text_input_ids'] = encoding['input_ids'].to(device)
        inputs['text_attention_mask'] = encoding['attention_mask'].to(device)

        detections = model.inference(inputs, score_threshold=args.threshold, topk=10)
        
        pred_boxes = detections[0]['boxes']
        pred_scores = detections[0]['scores']

        rgb_path = os.path.join(args.data, item.get('rgb', ''))
        if os.path.exists(rgb_path) and orig_size is not None:
            rgb_img = cv2.imread(rgb_path)
            
            vis_img = draw_boxes(rgb_img, pred_boxes, pred_scores, orig_size, args.threshold)
            
            ann = item.get('annotations', [])
            for a in ann:
                x, y, w, h = a['bbox']
                x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            save_path = os.path.join(args.output, f'sample_{idx:03d}.jpg')
            cv2.imwrite(save_path, vis_img)
            print(f"  Saved: {save_path}")
            print(f"  Text: {text}")
            print(f"  Boxes found: {len(pred_scores)}")
            if len(pred_scores) > 0:
                print(f"  Max score: {pred_scores.max():.3f}")
            
            results.append({
                'idx': idx,
                'text': text,
                'num_boxes': len(pred_scores),
                'max_score': float(pred_scores.max()) if len(pred_scores) > 0 else 0,
                'save_path': save_path
            })

    summary_path = os.path.join(args.output, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Visualization completed!")
    print(f"Results saved to: {args.output}")
    print(f"Summary: {summary_path}")
    print(f"\nSample results:")
    for r in results[:3]:
        print(f"  Sample {r['idx']}: {r['num_boxes']} boxes, max score: {r['max_score']:.3f}")
        print(f"    Text: {r['text']}")


if __name__ == '__main__':
    main()
