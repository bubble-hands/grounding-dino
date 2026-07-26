import sys
import os
import importlib.util

project_root = 'E:/eee/Grounding DINO'
sys.path.insert(0, project_root)

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

dataset_path = os.path.join(groundingdino_path, 'datasets', '__init__.py')
spec_dataset = importlib.util.spec_from_file_location('groundingdino.datasets', dataset_path)
dataset_module = importlib.util.module_from_spec(spec_dataset)
sys.modules['groundingdino.datasets'] = dataset_module
spec_dataset.loader.exec_module(dataset_module)

from groundingdino.config.defaults import _C
from groundingdino.datasets.dataset import MultiModalDataset

cfg = _C.clone()
cfg.DATASETS.DATA_PATH = './data'
cfg.MODEL.MULTI_MODAL.MODALITIES = ['rgb', 'ir', 'depth']

ds = MultiModalDataset(cfg, split='train')
print(f'Train dataset size: {len(ds)}')

item = ds[0]
print(f'Item keys: {list(item.keys())}')
if 'targets' in item:
    print(f'Target labels: {item["targets"]["labels"]}')
    print(f'Target boxes: {item["targets"]["boxes"]}')
