"""
分析训练数据中的坐标分布，诊断模型预测集中在左下方的原因。
"""
import json
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train.json")

def analyze_distribution():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    all_cx, all_cy, all_w, all_h = [], [], [], []
    for item in data:
        for ann in item.get("annotations", []):
            bbox = ann.get("bbox", [0, 0, 0, 0])
            if len(bbox) == 4:
                cx, cy, w, h = bbox
                all_cx.append(cx)
                all_cy.append(cy)
                all_w.append(w)
                all_h.append(h)
                
    if not all_cx:
        print("No annotations found.")
        return

    all_cx = np.array(all_cx)
    all_cy = np.array(all_cy)
    all_w = np.array(all_w)
    all_h = np.array(all_h)
    
    print(f"Total annotations: {len(all_cx)}")
    
    # 1. 统计中心点分布
    print("\n" + "="*60)
    print("1. 标注框中心点 (cx, cy) 分布统计")
    print("="*60)
    
    def print_stats(name, arr):
        print(f"{name}: min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}, median={np.median(arr):.4f}")
        
    print_stats("cx (中心X)", all_cx)
    print_stats("cy (中心Y)", all_cy)
    
    # 2. 计算落在各象限的比例 (图像被分为 4 块)
    print("\n" + "="*60)
    print("2. 目标框位于各象限的比例")
    print("="*60)
    
    # cx < 0.5 是左, cx > 0.5 是右
    # cy < 0.5 是上, cy > 0.5 是下
    left = all_cx < 0.5
    right = all_cx >= 0.5
    top = all_cy < 0.5
    bottom = all_cy >= 0.5
    
    print(f"  左半部分 (cx < 0.5): {left.sum()} ({left.mean()*100:.1f}%)")
    print(f"  右半部分 (cx >= 0.5): {right.sum()} ({right.mean()*100:.1f}%)")
    print(f"  上半部分 (cy < 0.5): {top.sum()} ({top.mean()*100:.1f}%)")
    print(f"  下半部分 (cy >= 0.5): {bottom.sum()} ({bottom.mean()*100:.1f}%)")
    
    print(f"  ↘ 左下象限 (cx<0.5, cy>=0.5): {(left & bottom).sum()} ({(left & bottom).mean()*100:.1f}%)")
    print(f"  ↙ 左上象限 (cx<0.5, cy<0.5): {(left & top).sum()} ({(left & top).mean()*100:.1f}%)")
    print(f"  ↗ 右上象限 (cx>=0.5, cy<0.5): {(right & top).sum()} ({(right & top).mean()*100:.1f}%)")
    print(f"  ↖ 右下象限 (cx>=0.5, cy>=0.5): {(right & bottom).sum()} ({(right & bottom).mean()*100:.1f}%)")
    
    # 3. 可视化分布直方图 (终端简单版)
    print("\n" + "="*60)
    print("3. 中心点坐标分布 (直方图, bins=10)")
    print("="*60)
    
    def print_histogram(name, arr):
        hist, edges = np.histogram(arr, bins=10, range=(0, 1))
        max_count = hist.max() if hist.max() > 0 else 1
        print(f"\n{name} 分布:")
        for i in range(len(hist)):
            bar_len = int(hist[i] / max_count * 40)
            bar = "#" * bar_len
            print(f"  [{edges[i]:.2f}-{edges[i+1]:.2f}]: {bar} {hist[i]} ({hist[i]/len(arr)*100:.1f}%)")
            
    print_histogram("cx (水平位置)", all_cx)
    print_histogram("cy (垂直位置)", all_cy)

    # 4. 检查 bbox 宽高分布
    print("\n" + "="*60)
    print("4. 框尺寸分布")
    print("="*60)
    print_stats("w (宽度)", all_w)
    print_stats("h (高度)", all_h)

if __name__ == "__main__":
    analyze_distribution()
