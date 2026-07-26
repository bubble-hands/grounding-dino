import os
import sys
import importlib.util

project_root = os.path.dirname(os.path.abspath(__file__))

print(f"Project root: {project_root}")
print(f"Changing current directory to project root...")
os.chdir(project_root)
print(f"Current directory: {os.getcwd()}")

sys.path.insert(0, '.')
print(f"sys.path[0]: {sys.path[0]}")

try:
    from groundingdino.config import get_cfg
    print("SUCCESS: groundingdino.config imported!")
    cfg = get_cfg()
    print(f"Config loaded: {cfg}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()