import os
import sys
import torch
import importlib.util

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def import_groundingdino():
    groundingdino_path = os.path.join(project_root, 'groundingdino')
    
    init_path = os.path.join(groundingdino_path, '__init__.py')
    spec = importlib.util.spec_from_file_location('groundingdino', init_path)
    groundingdino = importlib.util.module_from_spec(spec)
    sys.modules['groundingdino'] = groundingdino
    spec.loader.exec_module(groundingdino)
    
    config_path = os.path.join(groundingdino_path, 'config', '__init__.py')
    spec_config = importlib.util.spec_from_file_location('groundingdino.config', config_path)
    config_module = importlib.util.module_from_spec(spec_config)
    sys.modules['groundingdino.config'] = config_module
    spec_config.loader.exec_module(config_module)
    
    defaults_path = os.path.join(groundingdino_path, 'config', 'defaults.py')
    spec_defaults = importlib.util.spec_from_file_location('groundingdino.config.defaults', defaults_path)
    defaults_module = importlib.util.module_from_spec(spec_defaults)
    sys.modules['groundingdino.config.defaults'] = defaults_module
    spec_defaults.loader.exec_module(defaults_module)
    
    return defaults_module._C

cfg = import_groundingdino()
exec(open('groundingdino/config/GroundingDINO_SwinT_MultiModal.py').read())

print("=== Config ===")
print(f"INPUT.SIZE_TRAIN: {cfg.INPUT.SIZE_TRAIN}")
print(f"MODEL.BACKBONE.OUT_CHANNELS: {cfg.MODEL.BACKBONE.OUT_CHANNELS}")
print(f"MODEL.HIDDEN_DIM: {cfg.MODEL.HIDDEN_DIM}")
print(f"MODEL.TEXT_ENCODER.DIM: {cfg.MODEL.TEXT_ENCODER.DIM}")

from groundingdino.models.backbone import MultiModalBackbone

print("\n=== Testing Backbone ===")
backbone = MultiModalBackbone(cfg)
backbone.eval()

inputs = {
    'rgb': torch.randn(2, 3, 512, 512),
    'ir': torch.randn(2, 1, 512, 512),
    'depth': torch.randn(2, 1, 512, 512),
}

with torch.no_grad():
    features = backbone(inputs)

for i, f in enumerate(features):
    if f is not None:
        print(f"Level {i}: {f.shape}")
    else:
        print(f"Level {i}: None")

from groundingdino.models.feature_enhancer import FeatureEnhancer

print("\n=== Testing Feature Enhancer ===")
enhancer = FeatureEnhancer(cfg)
enhancer.eval()

text_input_ids = torch.randint(0, 1000, (2, 64))
text_attention_mask = torch.ones(2, 64)

with torch.no_grad():
    try:
        enhanced, text_feat = enhancer(features, text_input_ids, text_attention_mask)
        print("Enhancer passed!")
        for i, e in enumerate(enhanced):
            print(f"Enhanced Level {i}: {e.shape}")
    except Exception as e:
        print(f"Enhancer error: {e}")