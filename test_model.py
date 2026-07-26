import os
import sys
import torch

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from groundingdino.config.defaults import _C as cfg
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