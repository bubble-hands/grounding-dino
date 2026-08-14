import os
import json
import cv2
import numpy as np


def extract_color_features(crop_rgb):
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_BGR2HSV)
    
    avg_hue = np.mean(hsv[:, :, 0])
    avg_sat = np.mean(hsv[:, :, 1])
    avg_val = np.mean(hsv[:, :, 2])
    
    if avg_sat < 30:
        return "dark" if avg_val < 80 else "light"
    
    color_ranges = [
        ("red", ((0, 15), (170, 180))),
        ("orange", ((15, 30),)),
        ("yellow", ((30, 45),)),
        ("green", ((45, 75),)),
        ("cyan", ((75, 90),)),
        ("blue", ((90, 120),)),
        ("purple", ((120, 150),)),
        ("pink", ((150, 170),)),
    ]
    
    for color, ranges in color_ranges:
        for r in ranges:
            if avg_hue >= r[0] and avg_hue <= r[1]:
                return color
    
    return "colored"


def analyze_pose(crop_rgb, bbox):
    h, w = crop_rgb.shape[:2]
    
    if h > w * 2:
        return "standing"
    elif w > h * 1.5:
        return "lying"
    elif h > w * 1.5:
        return "walking"
    else:
        return "standing"


def analyze_scene_context(img_rgb, bbox):
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    height, width = img_rgb.shape[:2]
    center_x = x + w / 2
    center_y = y + h / 2
    
    context = []
    
    ground_region = img_rgb[max(0, int(height * 0.7)):, :]
    sky_region = img_rgb[:int(height * 0.3), :]
    
    ground_avg = np.mean(ground_region)
    sky_avg = np.mean(sky_region)
    
    if center_y > height * 0.7:
        if ground_avg < 80:
            context.append("on the road")
        elif ground_avg > 120:
            context.append("on the sidewalk")
        else:
            context.append("on the ground")
    elif center_y < height * 0.3:
        context.append("in the distance")
    else:
        context.append("in the middle ground")
    
    if center_x < width * 0.3:
        context.append("on the left")
    elif center_x > width * 0.7:
        context.append("on the right")
    
    return " ".join(context)


def analyze_object_type(crop_rgb, crop_ir, crop_depth):
    h, w = crop_rgb.shape[:2]
    
    aspect_ratio = w / h if h > 0 else 1
    area = w * h
    
    ir_std = np.std(crop_ir)
    depth_avg = np.mean(crop_depth[crop_depth > 0]) if np.any(crop_depth > 0) else 0
    
    if aspect_ratio > 2 and area > 10000:
        return "vehicle"
    elif aspect_ratio > 1.5 and area > 5000:
        return "car"
    elif h > w * 1.5 and area > 2000:
        return "person"
    elif h > w * 1.2 and area > 1000:
        return "human"
    elif ir_std > 40:
        return "heat source"
    elif depth_avg > 15000:
        return "distant object"
    elif area > 50000:
        return "large object"
    elif area < 2000:
        return "small object"
    else:
        return "object"


def generate_semantic_description(img_rgb, img_ir, img_depth, bbox):
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    height, width = img_rgb.shape[:2]
    
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    
    if x2 <= x1 or y2 <= y1:
        return "a target object"
    
    crop_rgb = img_rgb[y1:y2, x1:x2]
    crop_ir = img_ir[y1:y2, x1:x2]
    crop_depth = img_depth[y1:y2, x1:x2]
    
    obj_type = analyze_object_type(crop_rgb, crop_ir, crop_depth)
    color = extract_color_features(crop_rgb)
    pose = analyze_pose(crop_rgb, bbox)
    context = analyze_scene_context(img_rgb, bbox)
    
    article = "an" if obj_type[0] in "aeiouAEIOU" else "a"
    
    descriptions = [
        f"{article} {color} {obj_type}",
        f"{article} {color} {obj_type} {pose}",
        f"{article} {color} {obj_type} {pose} {context}",
        f"{article} {obj_type} {pose}",
        f"{article} {color} {obj_type} {context}",
        f"{article} {obj_type} {context}",
        f"{article} {color} {obj_type} standing",
        f"{article} {obj_type} in the scene",
    ]
    
    complexity = 0
    if len(color) > 4:
        complexity += 1
    if pose != "standing":
        complexity += 1
    if len(context) > 10:
        complexity += 1
    
    if complexity >= 2:
        idx = 2
    elif complexity == 1:
        idx = 1
    else:
        idx = 0
    
    return descriptions[idx]


def update_dataset_with_semantic_text(dataset_path, json_file):
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
            text = generate_semantic_description(img_rgb, img_ir, img_depth, bbox)
            sample['text'] = text
    
    output_file = json_file.replace('.json', '_semantic.json')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Updated dataset saved to {output_file}")
    
    text_distribution = {}
    for sample in data:
        text = sample['text']
        text_distribution[text] = text_distribution.get(text, 0) + 1
    
    print("\nSemantic text distribution (top 30):")
    for text, count in sorted(text_distribution.items(), key=lambda x: -x[1])[:30]:
        print(f"  {text}: {count}")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, 'data')
    
    train_json = os.path.join(dataset_path, 'train.json')
    val_json = os.path.join(dataset_path, 'val.json')
    
    if os.path.exists(train_json):
        update_dataset_with_semantic_text(dataset_path, train_json)
    
    if os.path.exists(val_json):
        update_dataset_with_semantic_text(dataset_path, val_json)


if __name__ == '__main__':
    main()