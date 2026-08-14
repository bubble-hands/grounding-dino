import os
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_json = os.path.join(project_root, 'data', 'train.json')

with open(train_json, 'r') as f:
    data = json.load(f)

print('示例语义描述：')
for i, s in enumerate(data[:15]):
    print(f'{i+1}. {s["text"]}')