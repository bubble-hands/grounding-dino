import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
print(f"Project root: {project_root}")

sys.path.insert(0, project_root)
print(f"\nsys.path[0]: {sys.path[0]}")

groundingdino_path = os.path.join(project_root, 'groundingdino')
print(f"\ngroundingdino path: {groundingdino_path}")
print(f"Is directory: {os.path.isdir(groundingdino_path)}")

init_path = os.path.join(groundingdino_path, '__init__.py')
print(f"\n__init__.py exists: {os.path.exists(init_path)}")
if os.path.exists(init_path):
    with open(init_path, 'r') as f:
        print(f"__init__.py content: {repr(f.read())}")

version_path = os.path.join(groundingdino_path, 'version.py')
print(f"\nversion.py exists: {os.path.exists(version_path)}")

config_path = os.path.join(groundingdino_path, 'config', '__init__.py')
print(f"\nconfig/__init__.py exists: {os.path.exists(config_path)}")

try:
    import importlib.util
    spec = importlib.util.find_spec('groundingdino')
    print(f"\nimportlib spec: {spec}")
    
    if spec is None:
        print("Trying to find in sys.path...")
        for p in sys.path[:5]:
            full_path = os.path.join(p, 'groundingdino')
            print(f"  {p}: {os.path.exists(full_path)}")
            
except Exception as e:
    print(f"\nError: {e}")