import json

with open(r'E:\eee\Grounding DINO\data\train.json', 'r') as f:
    data = json.load(f)

print('示例语义描述：')
for i, s in enumerate(data[:15]):
    print(f'{i+1}. {s["text"]}')