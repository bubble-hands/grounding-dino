import os
import sys

print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

project_root = os.path.dirname(os.path.abspath(__file__))
print(f"Project root: {project_root}")
print(f"groundingdino exists: {os.path.exists(os.path.join(project_root, 'groundingdino'))}")

sys.path.insert(0, project_root)
print(f"\nAfter inserting into sys.path:")

try:
    from groundingdino.config import get_cfg
    print("SUCCESS: groundingdino.config imported!")
    cfg = get_cfg()
    print(f"Config loaded: {cfg}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()