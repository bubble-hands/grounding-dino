import os
import json
import cv2
import numpy as np


def analyze_target(img_rgb, img_ir, img_depth, bbox):
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    height, width = img_rgb.shape[:2]
    
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    
    if x2 <= x1 or y2 <= y1:
        return "target"
    
    crop_rgb = img_rgb[y1:y2, x1:x2]
    crop_ir = img_ir[y1:y2, x1:x2]
    crop_depth = img_depth[y1:y2, x1:x2]
    
    area = w * h
    aspect_ratio = w / h if h > 0 else 1
    
    depth_values = crop_depth[crop_depth > 0]
    avg_depth = np.mean(depth_values) if len(depth_values) > 0 else 0
    
    ir_avg = np.mean(crop_ir)
    ir_std = np.std(crop_ir)
    
    rgb_avg = np.mean(crop_rgb)
    rgb_std = np.std(crop_rgb)
    
    area_category = ""
    if area < 5000:
        area_category = "small "
    elif area > 50000:
        area_category = "large "
    
    depth_category = ""
    if avg_depth > 0:
        if avg_depth < 5000:
            depth_category = "nearby "
        elif avg_depth > 15000:
            depth_category = "distant "
    
    ir_category = ""
    if ir_std > 30:
        ir_category = "thermal "
    elif ir_avg > 150:
        ir_category = "warm "
    
    position_category = ""
    center_x = (x + w/2) / width
    center_y = (y + h/2) / height
    
    if center_y < 0.3:
        position_category = "upper "
    elif center_y > 0.7:
        position_category = "lower "
    
    if center_x < 0.3:
        position_category += "left "
    elif center_x > 0.7:
        position_category += "right "
    
    text_parts = []
    if area_category:
        text_parts.append(area_category)
    if depth_category:
        text_parts.append(depth_category)
    if ir_category:
        text_parts.append(ir_category)
    if position_category:
        text_parts.append(position_category)
    
    if aspect_ratio > 2:
        text_parts.append("horizontal")
    elif aspect_ratio < 0.5:
        text_parts.append("vertical")
    
    if not text_parts:
        text = "target object"
    else:
        text = " ".join(text_parts).strip() + " object"
    
    return text


def update_dataset_with_text(dataset_path, json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    total_samples = len(data)
    print(f"Processing {total_samples} samples from {json_file}...")
    
    for i, sample in enumerate(data):
        if i % 100 == 0:
            print(f"Processing sample {i}/{total_samples}")
        
        rgb_path = os.path.join(dataset_path, sample['rgb'])
        ir_path = os.path.join(dataset_path, sample['ir'])
        depth_path = os.path.join(dataset_path, sample['depth'])
        
        if not os.path.exists(rgb_path):
            print(f"Warning: {rgb_path} not found")
            continue
        if not os.path.exists(ir_path):
            print(f"Warning: {ir_path} not found")
            continue
        if not os.path.exists(depth_path):
            print(f"Warning: {depth_path} not found")
            continue
        
        img_rgb = cv2.imread(rgb_path)
        img_ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        img_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        if img_rgb is None or img_ir is None or img_depth is None:
            print(f"Warning: Failed to read images for sample {i}")
            continue
        
        for ann in sample['annotations']:
            bbox = ann['bbox']
            text = analyze_target(img_rgb, img_ir, img_depth, bbox)
            sample['text'] = text
    
    output_file = json_file.replace('.json', '_with_text.json')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Updated dataset saved to {output_file}")
    
    text_distribution = {}
    for sample in data:
        text = sample['text']
        text_distribution[text] = text_distribution.get(text, 0) + 1
    
    print("\nText prompt distribution:")
    for text, count in sorted(text_distribution.items(), key=lambda x: -x[1]):
        print(f"  {text}: {count} samples")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, 'data')
    
    train_json = os.path.join(dataset_path, 'train.json')
    val_json = os.path.join(dataset_path, 'val.json')
    
    if os.path.exists(train_json):
        update_dataset_with_text(dataset_path, train_json)
    
    if os.path.exists(val_json):
        update_dataset_with_text(dataset_path, val_json)


if __name__ == '__main__':
    main()