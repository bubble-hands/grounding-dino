import os
import json

data_path = 'E:/eee/Grounding DINO/data'
print(f"Data path exists: {os.path.exists(data_path)}")

if os.path.exists(data_path):
    files = os.listdir(data_path)
    print(f"Files in data directory: {files}")
    
    train_json = os.path.join(data_path, 'train.json')
    val_json = os.path.join(data_path, 'val.json')
    
    print(f"\ntrain.json exists: {os.path.exists(train_json)}")
    print(f"val.json exists: {os.path.exists(val_json)}")
    
    if os.path.exists(train_json):
        with open(train_json, 'r') as f:
            train_data = json.load(f)
        print(f"\nTrain data length: {len(train_data)}")
        if train_data:
            sample = train_data[0]
            print(f"Sample keys: {list(sample.keys())}")
            print(f"Sample text: {sample.get('text', '')[:50]}...")
            print(f"Sample rgb: {sample.get('rgb', '')}")
            print(f"Sample ir: {sample.get('ir', '')}")
            print(f"Sample depth: {sample.get('depth', '')}")
            if 'annotations' in sample:
                print(f"Sample annotations: {sample['annotations']}")