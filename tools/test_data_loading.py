"""
测试训练时的数据加载流程，检查坐标归一化是否正确。
模拟 MultiModalDataset 的完整加载过程。
"""
import os
import sys
import json
import importlib.util
import torch
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def import_groundingdino():
    """动态导入 groundingdino 包"""
    gdino_path = os.path.join(project_root, 'groundingdino')

    def _load(mod_name, sub_path):
        init_path = os.path.join(gdino_path, *sub_path, '__init__.py')
        spec = importlib.util.spec_from_file_location(mod_name, init_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return module

    _load('groundingdino', [])
    _load('groundingdino.config', ['config'])
    _load('groundingdino.models', ['models'])
    _load('groundingdino.datasets', ['datasets'])

    from groundingdino.config import get_cfg
    from groundingdino.datasets.dataset import MultiModalDataset
    return get_cfg, MultiModalDataset


def main():
    get_cfg, MultiModalDataset = import_groundingdino()
    cfg = get_cfg()

    # 加载训练配置
    config_path = os.path.join(project_root, 'groundingdino', 'config', 'GroundingDINO_Fused_Train.py')
    if os.path.exists(config_path):
        exec(open(config_path).read())
        cfg = get_cfg()
        print(f"Loaded config from: {config_path}")

    # 创建数据集
    print("\n" + "="*60)
    print("创建训练数据集")
    print("="*60)
    
    dataset = MultiModalDataset(cfg, split='train')
    
    print(f"数据集大小: {len(dataset)}")
    
    # 测试前几个样本
    print("\n" + "="*60)
    print("检查数据加载结果")
    print("="*60)
    
    all_target_boxes = []
    all_orig_sizes = []
    
    for idx in range(min(10, len(dataset))):
        try:
            inputs = dataset[idx]
            
            print(f"\n样本 {idx}:")
            print(f"  模态: {[k for k in inputs.keys() if k not in ['text', 'text_input_ids', 'text_attention_mask', 'targets']]}")
            
            # 检查各模态
            for mod in ['rgb', 'ir', 'depth']:
                if mod in inputs:
                    tensor = inputs[mod]
                    is_fallback = inputs.get(f'_{mod}_fallback', False)
                    print(f"  {mod}: shape={tensor.shape}, range=[{tensor.min():.4f}, {tensor.max():.4f}], fallback={is_fallback}")
            
            # 检查 targets
            if 'targets' in inputs:
                targets = inputs['targets']
                boxes = targets['boxes'].numpy()
                labels = targets['labels'].numpy()
                
                print(f"  targets: {len(boxes)} 个框")
                for i, (box, label) in enumerate(zip(boxes, labels)):
                    cx, cy, w, h = box
                    all_target_boxes.append(box)
                    print(f"    框 {i}: [{cx:.4f}, {cy:.4f}, {w:.4f}, {h:.4f}], label={label}")
            
            # 检查文本
            if 'text' in inputs:
                print(f"  text: {inputs['text'][:50]}...")
            if 'text_input_ids' in inputs:
                print(f"  text_input_ids: shape={inputs['text_input_ids'].shape}")
                
        except Exception as e:
            print(f"\n样本 {idx}: 加载失败 - {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总统计
    if all_target_boxes:
        all_target_boxes = np.array(all_target_boxes)
        
        print("\n" + "="*60)
        print("Target 框坐标分布统计 (归一化)")
        print("="*60)
        print(f"cx: min={all_target_boxes[:,0].min():.4f}, max={all_target_boxes[:,0].max():.4f}, mean={all_target_boxes[:,0].mean():.4f}")
        print(f"cy: min={all_target_boxes[:,1].min():.4f}, max={all_target_boxes[:,1].max():.4f}, mean={all_target_boxes[:,1].mean():.4f}")
        print(f"w: min={all_target_boxes[:,2].min():.4f}, max={all_target_boxes[:,2].max():.4f}, mean={all_target_boxes[:,2].mean():.4f}")
        print(f"h: min={all_target_boxes[:,3].min():.4f}, max={all_target_boxes[:,3].max():.4f}, mean={all_target_boxes[:,3].mean():.4f}")
        
        # 象限分布
        cx = all_target_boxes[:, 0]
        cy = all_target_boxes[:, 1]
        left = cx < 0.5
        right = cx >= 0.5
        top = cy < 0.5
        bottom = cy >= 0.5
        
        print(f"\n左下象限 (cx<0.5, cy>=0.5): {(left & bottom).sum()} ({(left & bottom).mean()*100:.1f}%)")
        print(f"左上象限 (cx<0.5, cy<0.5): {(left & top).sum()} ({(left & top).mean()*100:.1f}%)")
        print(f"右上象限 (cx>=0.5, cy<0.5): {(right & top).sum()} ({(right & top).mean()*100:.1f}%)")
        print(f"右下象限 (cx>=0.5, cy>=0.5): {(right & bottom).sum()} ({(right & bottom).mean()*100:.1f}%)")


if __name__ == '__main__':
    main()