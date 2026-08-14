import os
import json
import cv2
import numpy as np


def get_object_name(crop_rgb, crop_ir, bbox):
    h, w = crop_rgb.shape[:2]
    area = w * h
    aspect_ratio = w / h if h > 0 else 1
    
    if aspect_ratio > 2.5 and area > 15000:
        return np.random.choice(["vehicle", "car", "automobile", "truck", "SUV"])
    elif aspect_ratio > 1.5 and area > 5000:
        return np.random.choice(["car", "vehicle", "sedan", "automobile"])
    elif h > w * 1.5 and area > 3000:
        return np.random.choice(["person", "human", "individual", "man", "woman", "pedestrian"])
    elif h > w * 1.2 and area > 1500:
        return np.random.choice(["human", "person", "figure", "individual"])
    elif h > w * 2 and area > 5000:
        return np.random.choice(["pole", "signpost", "street sign", "traffic light"])
    elif area > 30000:
        return np.random.choice(["large object", "building", "structure", "wall"])
    elif area < 1500:
        return np.random.choice(["small object", "object", "item"])
    else:
        return np.random.choice(["object", "target", "thing"])


def get_color_description(crop_rgb):
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_BGR2HSV)
    avg_hue = np.mean(hsv[:, :, 0])
    avg_sat = np.mean(hsv[:, :, 1])
    avg_val = np.mean(hsv[:, :, 2])
    
    if avg_sat < 25:
        if avg_val < 50:
            return np.random.choice(["black", "dark", "dark-colored", "dark-clad"])
        elif avg_val > 180:
            return np.random.choice(["white", "light", "light-colored"])
        else:
            return np.random.choice(["gray", "grey", "neutral-colored"])
    else:
        if (avg_hue >= 0 and avg_hue <= 15) or (avg_hue >= 170 and avg_hue <= 180):
            return np.random.choice(["red", "red-colored", "reddish"])
        elif avg_hue > 15 and avg_hue <= 30:
            return np.random.choice(["orange", "orange-colored"])
        elif avg_hue > 30 and avg_hue <= 45:
            return np.random.choice(["yellow", "yellow-colored"])
        elif avg_hue > 45 and avg_hue <= 75:
            return np.random.choice(["green", "green-colored", "greenish"])
        elif avg_hue > 75 and avg_hue <= 90:
            return np.random.choice(["cyan", "cyan-colored", "teal"])
        elif avg_hue > 90 and avg_hue <= 120:
            return np.random.choice(["blue", "blue-colored", "bluish"])
        elif avg_hue > 120 and avg_hue <= 150:
            return np.random.choice(["purple", "purple-colored", "violet"])
        elif avg_hue > 150 and avg_hue <= 170:
            return np.random.choice(["pink", "pink-colored"])
    
    return "colored"


def get_action_or_pose(crop_rgb, bbox):
    h, w = crop_rgb.shape[:2]
    aspect_ratio = w / h if h > 0 else 1
    
    obj_name = get_object_name(crop_rgb, None, bbox)
    
    if "person" in obj_name or "human" in obj_name or "man" in obj_name or "woman" in obj_name:
        if h > w * 2:
            return np.random.choice(["standing", "standing upright", "standing still"])
        elif h > w * 1.3:
            return np.random.choice(["walking", "moving", "walking along"])
        elif w > h * 1.5:
            return np.random.choice(["lying down", "sitting", "kneeling"])
        else:
            return np.random.choice(["standing", "walking", "moving"])
    elif "car" in obj_name or "vehicle" in obj_name or "SUV" in obj_name:
        if aspect_ratio > 2:
            return np.random.choice(["parked", "stopped", "stationary"])
        else:
            return np.random.choice(["driving", "moving", "traveling"])
    else:
        return np.random.choice(["standing", "located", "positioned"])


def get_location_context(img_rgb, bbox):
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    height, width = img_rgb.shape[:2]
    center_x = x + w / 2
    center_y = y + h / 2
    
    contexts = []
    
    if center_y > height * 0.7:
        contexts.append(np.random.choice(["on the sidewalk", "on the road", "on the street"]))
    elif center_y > height * 0.4:
        contexts.append(np.random.choice(["in the street", "on the road", "in the middle of the road"]))
    else:
        contexts.append(np.random.choice(["in the distance", "far away", "in the background"]))
    
    if center_x < width * 0.25:
        contexts.append(np.random.choice(["on the left", "near the left side"]))
    elif center_x > width * 0.75:
        contexts.append(np.random.choice(["on the right", "near the right side"]))
    elif center_x > width * 0.4 and center_x < width * 0.6:
        contexts.append(np.random.choice(["in the center", "in the middle"]))
    
    return contexts


def get_additional_details(crop_ir, crop_depth, obj_name):
    details = []
    
    ir_avg = np.mean(crop_ir)
    if ir_avg > 180:
        details.append(np.random.choice(["with strong heat signature", "emitting heat"]))
    elif ir_avg > 150:
        details.append(np.random.choice(["with visible heat signature", "warm"]))
    elif ir_avg < 50:
        details.append(np.random.choice(["cool", "not emitting heat"]))
    
    depth_vals = crop_depth[crop_depth > 0]
    if len(depth_vals) > 0:
        depth_avg = np.mean(depth_vals)
        if depth_avg < 3000:
            details.append(np.random.choice(["very close", "nearby"]))
        elif depth_avg > 15000:
            details.append(np.random.choice(["far away", "at a distance"]))
    
    if "person" in obj_name or "human" in obj_name:
        details.append(np.random.choice(["wearing dark clothes", "wearing light clothes", "in casual clothing", ""]))
    
    return [d for d in details if d]


def generate_natural_description(img_rgb, img_ir, img_depth, bbox):
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    height, width = img_rgb.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    
    if x2 <= x1 or y2 <= y1:
        return "A target object in the scene."
    
    crop_rgb = img_rgb[y1:y2, x1:x2]
    crop_ir = img_ir[y1:y2, x1:x2]
    crop_depth = img_depth[y1:y2, x1:x2]
    
    obj_name = get_object_name(crop_rgb, crop_ir, bbox)
    color = get_color_description(crop_rgb)
    action = get_action_or_pose(crop_rgb, bbox)
    contexts = get_location_context(img_rgb, bbox)
    details = get_additional_details(crop_ir, crop_depth, obj_name)
    
    templates = [
        "A {color} {obj_name} {action} {context}.",
        "A {obj_name} in {color} clothing {action} {context}.",
        "{article} {obj_name} {action} {context} {details}.",
        "{article} {color} {obj_name} {action} on the {context}.",
        "A {color} {obj_name} {action} near the {context}.",
        "{article} {obj_name} {action} {context}.",
    ]
    
    article = "An" if obj_name[0] in "aeiouAEIOU" else "A"
    
    if contexts:
        context_str = contexts[0]
        if len(contexts) > 1:
            context_str = contexts[0] + " " + contexts[1]
    else:
        context_str = "in the scene"
    
    if details:
        detail_str = ", ".join(details)
    else:
        detail_str = ""
    
    template_idx = np.random.randint(len(templates))
    
    description = templates[template_idx].format(
        article=article,
        color=color,
        obj_name=obj_name,
        action=action,
        context=context_str,
        details=detail_str
    )
    
    description = description.replace("  ", " ").replace(". .", ".").strip()
    if description.endswith("."):
        description = description[:-1] + "."
    elif not description.endswith("."):
        description += "."
    
    return description.capitalize()


def update_dataset_with_final_text(dataset_path, json_file):
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
            sample['text'] = "A target object in the scene."
            continue
        
        img_rgb = cv2.imread(rgb_path)
        img_ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        img_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        if img_rgb is None or img_ir is None or img_depth is None:
            sample['text'] = "A target object in the scene."
            continue
        
        for ann in sample['annotations']:
            bbox = ann['bbox']
            text = generate_natural_description(img_rgb, img_ir, img_depth, bbox)
            sample['text'] = text
    
    output_file = json_file.replace('.json', '_final.json')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Updated dataset saved to {output_file}")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, 'data')
    
    train_json = os.path.join(dataset_path, 'train.json')
    val_json = os.path.join(dataset_path, 'val.json')
    
    if os.path.exists(train_json):
        update_dataset_with_final_text(dataset_path, train_json)
    
    if os.path.exists(val_json):
        update_dataset_with_final_text(dataset_path, val_json)


if __name__ == '__main__':
    main()