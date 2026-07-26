import os
import sys
import importlib.util

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

groundingdino_path = os.path.join(project_root, 'groundingdino')
init_path = os.path.join(groundingdino_path, '__init__.py')

print(f"Trying to import groundingdino directly...")

try:
    spec = importlib.util.spec_from_file_location('groundingdino', init_path)
    if spec is not None:
        groundingdino = importlib.util.module_from_spec(spec)
        sys.modules['groundingdino'] = groundingdino
        spec.loader.exec_module(groundingdino)
        print(f"SUCCESS: groundingdino module loaded")
        print(f"Version: {groundingdino.__version__}")
        
        config_path = os.path.join(groundingdino_path, 'config', '__init__.py')
        spec_config = importlib.util.spec_from_file_location('groundingdino.config', config_path)
        if spec_config is not None:
            config_module = importlib.util.module_from_spec(spec_config)
            sys.modules['groundingdino.config'] = config_module
            spec_config.loader.exec_module(config_module)
            print(f"SUCCESS: groundingdino.config loaded")
            print(f"Has get_cfg: {hasattr(config_module, 'get_cfg')}")
            
            if hasattr(config_module, 'get_cfg'):
                cfg = config_module.get_cfg()
                print(f"Config loaded successfully!")
    else:
        print("ERROR: spec is None")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()