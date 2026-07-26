import os
import json
import cv2
import numpy as np


def analyze_color_palette(crop_rgb):
    pixels = crop_rgb.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    
    if len(unique_colors) < 10:
        return "uniform color"
    
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_BGR2HSV)
    avg_sat = np.mean(hsv[:, :, 1])
    
    if avg_sat > 80:
        return "colorful"
    elif avg_sat > 40:
        return "moderately colored"
    else:
        return "muted colors"


def detect_background(img_rgb, bbox):
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    height, width = img_rgb.shape[:2]
    
    center_x = x + w / 2
    center_y = y + h / 2
    
    regions = []
    
    if center_y > height * 0.75:
        ground = img_rgb[max(0, int(height * 0.8)):, :]
        road_likelihood = np.mean(ground[:, :, 0] > 80) + np.mean(ground[:, :, 1] > 80) + np.mean(ground[:, :, 2] > 80)
        if road_likelihood > 2:
            regions.append("on the road")
        else:
            regions.append("on the sidewalk")
    elif center_y > height * 0.5:
        regions.append("in the street")
    else:
        regions.append("in the distance")
    
    if center_x < width * 0.25:
        regions.append("on the left side")
    elif center_x > width * 0.75:
        regions.append("on the right side")
    
    return " ".join(regions)


def analyze_object_details(crop_rgb, crop_ir, crop_depth):
    h, w = crop_rgb.shape[:2]
    area = w * h
    
    details = []
    
    if h > w * 1.5:
        if area > 5000:
            details.append("tall")
        elif area < 2000:
            details.append("short")
    
    if w > h * 1.5:
        details.append("wide")
    
    ir_avg = np.mean(crop_ir)
    if ir_avg > 180:
        details.append("with strong thermal signature")
    elif ir_avg > 150:
        details.append("with thermal signature")
    elif ir_avg < 50:
        details.append("cool")
    
    depth_vals = crop_depth[crop_depth > 0]
    if len(depth_vals) > 0:
        depth_std = np.std(depth_vals)
        if depth_std > 5000:
            details.append("with varying depth")
    
    edge_density = cv2.Canny(crop_rgb, 50, 150).mean()
    if edge_density > 30:
        details.append("with complex features")
    elif edge_density < 10:
        details.append("smooth")
    
    return " ".join(details)


def generate_detailed_description(img_rgb, img_ir, img_depth, bbox):
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    height, width = img_rgb.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    
    if x2 <= x1 or y2 <= y1:
        return "a target object in the scene"
    
    crop_rgb = img_rgb[y1:y2, x1:x2]
    crop_ir = img_ir[y1:y2, x1:x2]
    crop_depth = img_depth[y1:y2, x1:x2]
    
    aspect_ratio = w / h if h > 0 else 1
    area = w * h
    
    ir_avg = np.mean(crop_ir)
    ir_std = np.std(crop_ir)
    
    object_types = []
    
    if aspect_ratio > 2 and area > 15000:
        object_types.append("vehicle")
        object_types.append("car")
        object_types.append("automobile")
    elif aspect_ratio > 1.5 and area > 5000:
        object_types.append("car")
        object_types.append("vehicle")
    elif h > w * 1.5 and area > 3000:
        object_types.append("person")
        object_types.append("human")
        object_types.append("individual")
    elif h > w * 1.2 and area > 1500:
        object_types.append("human")
        object_types.append("person")
    elif ir_avg > 180 and ir_std > 40:
        object_types.append("heat source")
        object_types.append("thermal object")
    elif area > 30000:
        object_types.append("large object")
        object_types.append("big object")
    elif area < 1500:
        object_types.append("small object")
    else:
        object_types.append("object")
    
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_BGR2HSV)
    avg_hue = np.mean(hsv[:, :, 0])
    avg_sat = np.mean(hsv[:, :, 1])
    avg_val = np.mean(hsv[:, :, 2])
    
    colors = []
    if avg_sat < 20:
        if avg_val < 60:
            colors.append("dark")
            colors.append("black")
            colors.append("dark-colored")
        else:
            colors.append("light")
            colors.append("white")
            colors.append("light-colored")
    else:
        if (avg_hue >= 0 and avg_hue <= 15) or (avg_hue >= 170 and avg_hue <= 180):
            colors.append("red")
            colors.append("reddish")
        elif avg_hue > 15 and avg_hue <= 30:
            colors.append("orange")
            colors.append("orange-colored")
        elif avg_hue > 30 and avg_hue <= 45:
            colors.append("yellow")
            colors.append("yellowish")
        elif avg_hue > 45 and avg_hue <= 75:
            colors.append("green")
            colors.append("greenish")
        elif avg_hue > 75 and avg_hue <= 90:
            colors.append("cyan")
            colors.append("cyan-colored")
        elif avg_hue > 90 and avg_hue <= 120:
            colors.append("blue")
            colors.append("bluish")
        elif avg_hue > 120 and avg_hue <= 150:
            colors.append("purple")
            colors.append("purple-colored")
        elif avg_hue > 150 and avg_hue <= 170:
            colors.append("pink")
            colors.append("pinkish")
    
    poses = []
    if h > w * 2:
        poses.append("standing upright")
        poses.append("standing")
        poses.append("in upright position")
    elif w > h * 1.5:
        poses.append("lying down")
        poses.append("horizontal")
    elif h > w * 1.3:
        poses.append("walking")
        poses.append("in motion")
        poses.append("moving")
    else:
        poses.append("standing")
        poses.append("stationary")
    
    contexts = detect_background(img_rgb, bbox)
    
    detail_str = analyze_object_details(crop_rgb, crop_ir, crop_depth)
    
    templates = [
        "A {color} {obj_type} {pose} {context}.",
        "A {color} {obj_type} {pose} {context} {details}.",
        "A {obj_type} in {color} clothing {pose} {context}.",
        "A {color} {obj_type} {pose} on the {context}.",
        "An {obj_type} with {color} appearance {pose} {context}.",
        "A {color} {obj_type} {pose} near the {context}.",
        "A {color} {obj_type} standing {context}.",
        "A {obj_type} {pose} {context} with {color} features.",
    ]
    
    color = colors[0] if colors else "colored"
    obj_type = object_types[0] if object_types else "object"
    pose = poses[0] if poses else "standing"
    
    template_idx = np.random.randint(len(templates))
    description = templates[template_idx].format(
        color=color,
        obj_type=obj_type,
        pose=pose,
        context=contexts,
        details=detail_str
    )
    
    description = description.replace("  ", " ").strip()
    if description.endswith("."):
        description = description[:-1] + "."
    
    return description.capitalize()


def update_dataset_with_advanced_text(dataset_path, json_file):
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
        
        if not os.path.exists(rgb_path) or not os.path.exists(ir_path) or not os.path.exists(depth_path):
            continue
        
        img_rgb = cv2.imread(rgb_path)
        img_ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        img_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        if img_rgb is None or img_ir is None or img_depth is None:
            continue
        
        for ann in sample['annotations']:
            bbox = ann['bbox']
            text = generate_detailed_description(img_rgb, img_ir, img_depth, bbox)
            sample['text'] = text
    
    output_file = json_file.replace('.json', '_advanced.json')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Updated dataset saved to {output_file}")


def main():
    dataset_path = r'E:\eee\Grounding DINO\data'
    
    train_json = os.path.join(dataset_path, 'train.json')
    val_json = os.path.join(dataset_path, 'val.json')
    
    if os.path.exists(train_json):
        update_dataset_with_advanced_text(dataset_path, train_json)
    
    if os.path.exists(val_json):
        update_dataset_with_advanced_text(dataset_path, val_json)


if __name__ == '__main__':
    main()