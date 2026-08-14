"""
检查数据加载流程是否正确，特别是 orig_size 的获取。
"""
import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train.json")

def check_data_loading():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")
    
    # 1. 检查前几张图像是否能正常加载
    print("\n=== 1. 测试图像加载 (前5条) ===")
    for i, item in enumerate(data[:5]):
        print(f"\n样本 {i}: {item['query_id']}")
        for mod in ['rgb', 'ir', 'depth']:
            img_path = item.get(mod)
            if img_path:
                exists = os.path.exists(img_path)
                if exists:
                    try:
                        from PIL import Image
                        img = Image.open(img_path)
                        print(f"  {mod}: 存在, 尺寸={img.size}, 模式={img.mode}")
                    except Exception as e:
                        print(f"  {mod}: 存在但加载失败 - {e}")
                else:
                    print(f"  {mod}: 不存在 - {img_path}")

    # 2. 检查数据中所有不同的 img_size
    print("\n=== 2. 检查数据中所有 img_size ===")
    all_sizes = {}
    for item in data:
        for ann in item.get('annotations', []):
            size = tuple(ann.get('img_size', [0, 0]))
            if size not in all_sizes:
                all_sizes[size] = 0
            all_sizes[size] += 1

    for size in sorted(all_sizes.keys()):
        print(f"  img_size={size}: {all_sizes[size]} 个标注")

    # 3. 模拟 _prepare_targets 计算
    print("\n=== 3. 模拟 _prepare_targets 归一化 (前10条) ===")
    for i, item in enumerate(data[:10]):
        ann = item['annotations'][0]
        x, y, w, h = ann['bbox']
        orig_w, orig_h = ann['img_size']
        
        # 检查 bbox 是否在图像范围内
        if x < 0 or y < 0 or x + w > orig_w or y + h > orig_h:
            print(f"  [警告] 样本 {item['query_id']}: bbox 超出图像范围!")
            print(f"    bbox=[{x}, {y}, {w}, {h}], img_size=[{orig_w}, {orig_h}]")
        
        cx = (x + w/2) / orig_w
        cy = (y + h/2) / orig_h
        nw = w / orig_w
        nh = h / orig_h
        
        if i < 3:  # 只打印前3条
            print(f"  {item['query_id']}: bbox=[{x}, {y}, {w}, {h}] -> normalized=[{cx:.4f}, {cy:.4f}, {nw:.4f}, {nh:.4f}]")

    # 4. 统计归一化坐标分布
    print("\n=== 4. 归一化坐标分布统计 ===")
    all_cx, all_cy = [], []
    for item in data:
        for ann in item.get('annotations', []):
            x, y, w, h = ann['bbox']
            orig_w, orig_h = ann['img_size']
            cx = (x + w/2) / orig_w
            cy = (y + h/2) / orig_h
            all_cx.append(cx)
            all_cy.append(cy)

    import numpy as np
    all_cx = np.array(all_cx)
    all_cy = np.array(all_cy)
    
    print(f"cx: min={all_cx.min():.4f}, max={all_cx.max():.4f}, mean={all_cx.mean():.4f}")
    print(f"cy: min={all_cy.min():.4f}, max={all_cy.max():.4f}, mean={all_cy.mean():.4f}")
    
    # 象限分布
    left = all_cx < 0.5
    right = all_cx >= 0.5
    top = all_cy < 0.5
    bottom = all_cy >= 0.5
    
    print(f"\n左下象限 (cx<0.5, cy>=0.5): {(left & bottom).sum()} ({(left & bottom).mean()*100:.1f}%)")
    print(f"左上象限 (cx<0.5, cy<0.5): {(left & top).sum()} ({(left & top).mean()*100:.1f}%)")
    print(f"右上象限 (cx>=0.5, cy<0.5): {(right & top).sum()} ({(right & top).mean()*100:.1f}%)")
    print(f"右下象限 (cx>=0.5, cy>=0.5): {(right & bottom).sum()} ({(right & bottom).mean()*100:.1f}%)")

if __name__ == '__main__':
    check_data_loading()