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

from groundingdino.models.groundingdino import GroundingDINO

print("Creating model...")
model = GroundingDINO(cfg)
model.eval()
print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

print("\nTesting forward pass...")
try:
    inputs = {
        'rgb': torch.randn(2, 3, 512, 512),
        'ir': torch.randn(2, 1, 512, 512),
        'depth': torch.randn(2, 1, 512, 512),
        'text_input_ids': torch.randint(0, 1000, (2, 64)),
        'text_attention_mask': torch.ones(2, 64),
    }
    
    with torch.no_grad():
        outputs = model(inputs)
    
    print(f"Forward pass successful!")
    print(f"pred_logits shape: {outputs['pred_logits'].shape}")
    print(f"pred_boxes shape: {outputs['pred_boxes'].shape}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()