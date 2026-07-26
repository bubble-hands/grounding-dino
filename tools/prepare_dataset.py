import os
import json
import shutil
import random
from pathlib import Path


def main():
    source_dir = r'E:\eee\Grounding DINO\Train-001'
    target_dir = r'E:\eee\Grounding DINO\data'
    
    train_ratio = 0.8
    val_ratio = 0.2
    
    scene_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d)) and d.isdigit()]
    scene_dirs.sort(key=int)
    
    print(f"发现 {len(scene_dirs)} 个场景文件夹")
    
    train_data = []
    val_data = []
    
    for scene_id in scene_dirs:
        scene_path = os.path.join(source_dir, scene_id)
        
        gt_file = os.path.join(scene_path, 'groundtruth.txt')
        if not os.path.exists(gt_file):
            print(f"警告: {scene_path} 缺少 groundtruth.txt，跳过")
            continue
        
        color_dir = os.path.join(scene_path, 'color')
        infrared_dir = os.path.join(scene_path, 'infrared')
        depth_dir = os.path.join(scene_path, 'depth')
        
        with open(gt_file, 'r') as f:
            lines = f.readlines()
        
        scene_samples = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            img_name = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            
            color_path = os.path.join(color_dir, img_name)
            ir_path = os.path.join(infrared_dir, img_name)
            depth_path = os.path.join(depth_dir, img_name)
            
            if not os.path.exists(color_path):
                print(f"警告: {color_path} 不存在")
                continue
            if not os.path.exists(ir_path):
                print(f"警告: {ir_path} 不存在")
                continue
            if not os.path.exists(depth_path):
                print(f"警告: {depth_path} 不存在")
                continue
            
            sample = {
                'scene_id': scene_id,
                'img_name': img_name,
                'color_path': color_path,
                'ir_path': ir_path,
                'depth_path': depth_path,
                'annotations': [{
                    'category_id': 0,
                    'bbox': [x, y, w, h]
                }]
            }
            scene_samples.append(sample)
        
        random.shuffle(scene_samples)
        split_idx = int(len(scene_samples) * train_ratio)
        train_data.extend(scene_samples[:split_idx])
        val_data.extend(scene_samples[split_idx:])
    
    print(f"\n训练集: {len(train_data)} 样本")
    print(f"验证集: {len(val_data)} 样本")
    
    train_rgb_dir = os.path.join(target_dir, 'train', 'rgb')
    train_ir_dir = os.path.join(target_dir, 'train', 'ir')
    train_depth_dir = os.path.join(target_dir, 'train', 'depth')
    
    val_rgb_dir = os.path.join(target_dir, 'val', 'rgb')
    val_ir_dir = os.path.join(target_dir, 'val', 'ir')
    val_depth_dir = os.path.join(target_dir, 'val', 'depth')
    
    for dir_path in [train_rgb_dir, train_ir_dir, train_depth_dir, val_rgb_dir, val_ir_dir, val_depth_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    train_json = []
    for i, sample in enumerate(train_data):
        new_name = f'train_{i:06d}.png'
        
        shutil.copy(sample['color_path'], os.path.join(train_rgb_dir, new_name))
        shutil.copy(sample['ir_path'], os.path.join(train_ir_dir, new_name))
        shutil.copy(sample['depth_path'], os.path.join(train_depth_dir, new_name))
        
        train_json.append({
            'rgb': f'train/rgb/{new_name}',
            'ir': f'train/ir/{new_name}',
            'depth': f'train/depth/{new_name}',
            'text': 'target',
            'annotations': sample['annotations']
        })
    
    val_json = []
    for i, sample in enumerate(val_data):
        new_name = f'val_{i:06d}.png'
        
        shutil.copy(sample['color_path'], os.path.join(val_rgb_dir, new_name))
        shutil.copy(sample['ir_path'], os.path.join(val_ir_dir, new_name))
        shutil.copy(sample['depth_path'], os.path.join(val_depth_dir, new_name))
        
        val_json.append({
            'rgb': f'val/rgb/{new_name}',
            'ir': f'val/ir/{new_name}',
            'depth': f'val/depth/{new_name}',
            'text': 'target',
            'annotations': sample['annotations']
        })
    
    with open(os.path.join(target_dir, 'train.json'), 'w') as f:
        json.dump(train_json, f, indent=2)
    
    with open(os.path.join(target_dir, 'val.json'), 'w') as f:
        json.dump(val_json, f, indent=2)
    
    print(f"\n数据集整理完成！")
    print(f"训练集JSON: {os.path.join(target_dir, 'train.json')}")
    print(f"验证集JSON: {os.path.join(target_dir, 'val.json')}")
    print(f"训练图像目录: {train_rgb_dir}, {train_ir_dir}, {train_depth_dir}")
    print(f"验证图像目录: {val_rgb_dir}, {val_ir_dir}, {val_depth_dir}")


if __name__ == '__main__':
    main()