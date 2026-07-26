import os
import sys
import importlib.util

project_root = os.path.dirname(os.path.abspath(__file__))

groundingdino_path = os.path.join(project_root, 'groundingdino')
config_path = os.path.join(groundingdino_path, 'config', '__init__.py')

print(f"Trying to import from {config_path}")

spec = importlib.util.spec_from_file_location('groundingdino.config', config_path)
config_module = importlib.util.module_from_spec(spec)
sys.modules['groundingdino.config'] = config_module
spec.loader.exec_module(config_module)

print(f"Has get_cfg: {hasattr(config_module, 'get_cfg')}")

if hasattr(config_module, 'get_cfg'):
    cfg = config_module.get_cfg()
    print(f"Config loaded successfully!")
    print(f"DATA_PATH: {cfg.DATASETS.DATA_PATH}")