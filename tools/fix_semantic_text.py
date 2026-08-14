import os
import json
import cv2
import numpy as np


def fix_article(text):
    text = text.replace("an blue", "a blue")
    text = text.replace("an cyan", "a cyan")
    text = text.replace("an green", "a green")
    text = text.replace("an purple", "a purple")
    text = text.replace("an yellow", "a yellow")
    text = text.replace("an orange", "an orange")
    text = text.replace("an red", "a red")
    text = text.replace("an pink", "a pink")
    return text


def enhance_description(img_rgb, img_ir, img_depth, bbox, text):
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    height, width = img_rgb.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    
    if x2 <= x1 or y2 <= y1:
        return text
    
    crop_rgb = img_rgb[y1:y2, x1:x2]
    crop_ir = img_ir[y1:y2, x1:x2]
    
    ir_avg = np.mean(crop_ir)
    ir_std = np.std(crop_ir)
    
    thermal_descriptions = []
    if ir_avg > 180 and ir_std > 30:
        thermal_descriptions.append("with strong heat signature")
    elif ir_avg > 150:
        thermal_descriptions.append("with visible heat signature")
    elif ir_avg < 50:
        thermal_descriptions.append("cool object")
    
    depth_avg = np.mean(img_depth[y1:y2, x1:x2][img_depth[y1:y2, x1:x2] > 0])
    if depth_avg > 0:
        if depth_avg < 3000:
            thermal_descriptions.append("very close")
        elif depth_avg < 8000:
            thermal_descriptions.append("close by")
        elif depth_avg > 15000:
            thermal_descriptions.append("far away")
    
    if thermal_descriptions:
        text = text.replace(" standing", " standing " + ", ".join(thermal_descriptions))
        text = text.replace(" walking", " walking " + ", ".join(thermal_descriptions))
        text = text.replace(" lying", " lying " + ", ".join(thermal_descriptions))
        if "standing" not in text and "walking" not in text and "lying" not in text:
            text = text + " " + ", ".join(thermal_descriptions)
    
    return text


def update_dataset_with_fixed_text(dataset_path, json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    total_samples = len(data)
    print(f"Processing {total_samples} samples from {json_file}...")
    
    for i, sample in enumerate(data):
        if i % 100 == 0:
            print(f"Processing sample {i}/{total_samples}")
        
        sample['text'] = fix_article(sample['text'])
        
        rgb_path = os.path.join(dataset_path, sample['rgb'])
        ir_path = os.path.join(dataset_path, sample['ir'])
        depth_path = os.path.join(dataset_path, sample['depth'])
        
        if os.path.exists(rgb_path) and os.path.exists(ir_path) and os.path.exists(depth_path):
            img_rgb = cv2.imread(rgb_path)
            img_ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
            img_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            
            if img_rgb is not None and img_ir is not None and img_depth is not None:
                for ann in sample['annotations']:
                    bbox = ann['bbox']
                    sample['text'] = enhance_description(img_rgb, img_ir, img_depth, bbox, sample['text'])
    
    output_file = json_file.replace('_semantic.json', '_semantic_enhanced.json')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Updated dataset saved to {output_file}")
    
    text_distribution = {}
    for sample in data:
        text = sample['text']
        text_distribution[text] = text_distribution.get(text, 0) + 1
    
    print("\nEnhanced text distribution (top 20):")
    for text, count in sorted(text_distribution.items(), key=lambda x: -x[1])[:20]:
        print(f"  {text}: {count}")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, 'data')
    
    train_json = os.path.join(dataset_path, 'train_semantic.json')
    val_json = os.path.join(dataset_path, 'val_semantic.json')
    
    if os.path.exists(train_json):
        update_dataset_with_fixed_text(dataset_path, train_json)
    
    if os.path.exists(val_json):
        update_dataset_with_fixed_text(dataset_path, val_json)


if __name__ == '__main__':
    main()