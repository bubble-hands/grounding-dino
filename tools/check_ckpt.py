import os
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ckpt_path = os.path.join(project_root, 'output', 'latest_checkpoint.pth')
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
print('Checkpoint epoch:', ckpt.get('epoch', '?'))
print('Best loss:', ckpt.get('best_loss', '?'))
print('Keys:', list(ckpt.keys()))
