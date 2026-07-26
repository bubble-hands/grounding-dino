import json

with open(r'E:\eee\Grounding DINO\data\train.json', 'r') as f:
    data = json.load(f)

print('高级语义描述示例：')
for i, s in enumerate(data[:20]):
    print(f'{i+1}. {s["text"]}')