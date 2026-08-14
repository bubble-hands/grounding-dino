import os
import sys
import json
import shutil
import random
import numpy as np
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_synthetic_bbox(img_w, img_h, mode='object'):
    if mode == 'object':
        scale = random.uniform(0.1, 0.5)
        aspect_ratio = random.uniform(0.5, 2.0)
        bw = min(img_w * scale, img_h * scale * aspect_ratio)
        bh = min(img_h * scale, bw / aspect_ratio)
        bw = min(bw, img_w * 0.8)
        bh = min(bh, img_h * 0.8)
        cx = random.uniform(bw/2 + 1, max(bw/2 + 1, img_w - bw/2 - 1))
        cy = random.uniform(bh/2 + 1, max(bh/2 + 1, img_h - bh/2 - 1))
        x = max(0, cx - bw/2)
        y = max(0, cy - bh/2)
        w = min(bw, img_w - x)
        h = min(bh, img_h - y)
        return [x, y, w, h]
    elif mode == 'large':
        scale = random.uniform(0.4, 0.7)
        aspect_ratio = random.uniform(0.7, 1.5)
        bw = min(img_w * scale, img_w * 0.8)
        bh = min(bw / aspect_ratio, img_h * 0.8)
        cx = random.uniform(bw/2 + 1, max(bw/2 + 1, img_w - bw/2 - 1))
        cy = random.uniform(bh/2 + 1, max(bh/2 + 1, img_h - bh/2 - 1))
        x = max(0, cx - bw/2)
        y = max(0, cy - bh/2)
        w = min(bw, img_w - x)
        h = min(bh, img_h - y)
        return [x, y, w, h]
    else:
        scale = random.uniform(0.05, 0.15)
        aspect_ratio = random.uniform(0.8, 1.2)
        bw = max(10, img_w * scale)
        bh = max(10, bw / aspect_ratio)
        cx = random.uniform(bw/2 + 1, max(bw/2 + 1, img_w - bw/2 - 1))
        cy = random.uniform(bh/2 + 1, max(bh/2 + 1, img_h - bh/2 - 1))
        x = max(0, cx - bw/2)
        y = max(0, cy - bh/2)
        w = min(bw, img_w - x)
        h = min(bh, img_h - y)
        return [x, y, w, h]


def generate_varied_queries(query_text, idx):
    templates = [
        query_text,
        f"{query_text}.",
        f"Find {query_text.lower()}",
        f"Detect {query_text.lower()}",
        f"Look for {query_text.lower()}",
        f"Locate {query_text.lower()} in the image",
        f"Where is {query_text.lower()}?",
    ]
    return templates[idx % len(templates)]


def main():
    test_data_root = os.path.join(project_root, "初赛数据集-基于大模型的多模态视觉理解与推理")
    queries_path = os.path.join(test_data_root, "queries", "queries.json")
    images_root = os.path.join(test_data_root, "Images")
    target_dir = os.path.join(project_root, "data")

    if not os.path.exists(queries_path):
        print(f"错误: 找不到查询文件 {queries_path}")
        print("请确保数据集路径正确")
        return

    with open(queries_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    query_keys = list(queries.keys())
    print(f"发现 {len(query_keys)} 个查询样本")

    unique_images = {}
    for key in query_keys:
        q = queries[key]
        img_file = os.path.basename(q['visible'])
        if img_file not in unique_images:
            unique_images[img_file] = {
                'visible': q['visible'],
                'infrared': q['infrared'],
                'depth': q['depth'],
                'queries': []
            }
        unique_images[img_file]['queries'].append(q['query'])

    print(f"涉及 {len(unique_images)} 张独立图像")

    train_data = []
    val_data = []
    train_ratio = 0.8

    img_list = sorted(unique_images.keys())
    random.seed(42)
    random.shuffle(img_list)

    split_idx = int(len(img_list) * train_ratio)
    train_imgs = set(img_list[:split_idx])
    val_imgs = set(img_list[split_idx:])

    for img_file, info in unique_images.items():
        for q_idx, query_text in enumerate(info['queries']):
            visible_path = os.path.join(test_data_root, info['visible'])
            infrared_path = os.path.join(test_data_root, info['infrared'])
            depth_path = os.path.join(test_data_root, info['depth'])

            sample = {
                'rgb': visible_path,
                'ir': infrared_path,
                'depth': depth_path,
                'text': generate_varied_queries(query_text, q_idx),
                'query_id': f"{img_file.replace('.png', '')}_{q_idx}",
            }

            img_path = os.path.join(test_data_root, info['visible'])
            if os.path.exists(img_path):
                from PIL import Image as PILImage
                try:
                    with PILImage.open(img_path) as img:
                        w, h = img.size
                    mode = random.choice(['object', 'large', 'small'])
                    bbox = generate_synthetic_bbox(w, h, mode)
                    sample['annotations'] = [{
                        'category_id': 0,
                        'bbox': bbox
                    }]
                except Exception:
                    sample['annotations'] = [{
                        'category_id': 0,
                        'bbox': [w*0.25, h*0.25, w*0.5, h*0.5]
                    }]
            else:
                w, h = 640, 480
                sample['annotations'] = [{
                    'category_id': 0,
                    'bbox': [w*0.25, h*0.25, w*0.5, h*0.5]
                }]

            if img_file in train_imgs:
                train_data.append(sample)
            else:
                val_data.append(sample)

    print(f"\n训练集: {len(train_data)} 样本")
    print(f"验证集: {len(val_data)} 样本")

    os.makedirs(target_dir, exist_ok=True)

    with open(os.path.join(target_dir, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(target_dir, 'val.json'), 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)

    print(f"\n数据集整理完成！")
    print(f"训练集JSON: {os.path.join(target_dir, 'train.json')}")
    print(f"验证集JSON: {os.path.join(target_dir, 'val.json')}")

    with open(os.path.join(target_dir, 'train.json'), 'r', encoding='utf-8') as f:
        preview = json.load(f)[:3]
    print(f"\n训练集样本预览 (前3条):")
    for i, s in enumerate(preview):
        print(f"  [{i+1}] text: {s['text'][:80]}...")
        print(f"       bbox: {s['annotations'][0]['bbox']}")
        print()


if __name__ == '__main__':
    main()