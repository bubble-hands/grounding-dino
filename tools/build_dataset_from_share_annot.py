"""
从 share_annot 的 approved.json 划分 + Train-001 原始图像，构建模型训练用的数据集。

输入:
  - share_annot_ac72f1d926bb2d23/share_annot_ac72f1d926bb2d23/{train,val}/approved.json
    · data: dict, key="{seq}_{frame}", value={visible,infrared,depth,bbox,width,height,query}
    · bbox 为归一化 [x1,y1,x2,y2]
  - Train-001/{seq}/{color,infrared,depth}/{frame}.png  (原始图像, 不修改)

输出:
  - data/train.json  (2875 条)
  - data/val.json    (719 条)

每条记录格式 (适配 MultiModalDataset):
  {
    "query_id": "001_00000001",
    "rgb":    "<abs>/Train-001/001/color/00000001.png",
    "ir":     "<abs>/Train-001/001/infrared/00000001.png",
    "depth":  "<abs>/Train-001/001/depth/00000001.png",
    "text":   "The white umbrella ...",
    "annotations": [{"category_id": 0, "bbox": [x,y,w,h 绝对像素], "img_size": [W,H]}],
    "source": "share_annot"
  }
"""
import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN001 = os.path.join(PROJECT_ROOT, 'Train-001')
SHARE_DIR = os.path.join(PROJECT_ROOT, 'share_annot_ac72f1d926bb2d23',
                         'share_annot_ac72f1d926bb2d23')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')


def remap_image_path(p):
    """share_annot 相对路径 -> Train-001 绝对路径
    visible:   Train/001/color/00000001.png        -> Train-001/001/color/00000001.png
    infrared:  Train/001/infrared/00000001.png     -> Train-001/001/infrared/00000001.png
    depth:     Processed/Train/001/depth_jet/00000001.png -> Train-001/001/depth/00000001.png
    """
    p = p.replace('Processed/Train/', 'Train/').replace('depth_jet', 'depth')
    p = p.replace('Train/', 'Train-001/', 1)
    return os.path.normpath(os.path.join(PROJECT_ROOT, p.replace('/', os.sep)))


def convert_split(split):
    src = os.path.join(SHARE_DIR, split, 'approved.json')
    with open(src, 'r', encoding='utf-8') as f:
        doc = json.load(f)

    meta = doc['metadata']
    data = doc['data']
    print(f'[{split}] metadata: sample_count={meta["sample_count"]} '
          f'sequence_count={meta["sequence_count"]}  data_len={len(data)}')

    records = []
    missing = 0
    for key, v in data.items():
        rgb_path = remap_image_path(v['visible'])
        ir_path = remap_image_path(v['infrared'])
        depth_path = remap_image_path(v['depth'])

        # 校验图片存在
        if not (os.path.exists(rgb_path) and os.path.exists(ir_path)
                and os.path.exists(depth_path)):
            missing += 1
            continue

        W, H = int(v['width']), int(v['height'])
        x1, y1, x2, y2 = v['bbox']
        # 归一化 [x1,y1,x2,y2] -> 绝对像素 [x,y,w,h]
        x = x1 * W
        y = y1 * H
        w = (x2 - x1) * W
        h = (y2 - y1) * H
        # 安全裁剪
        x = max(0.0, min(x, W - 1))
        y = max(0.0, min(y, H - 1))
        w = max(1.0, min(w, W - x))
        h = max(1.0, min(h, H - y))

        record = {
            'query_id': key,
            'rgb': rgb_path,
            'ir': ir_path,
            'depth': depth_path,
            'text': v['query'],
            'annotations': [{
                'category_id': 0,
                'bbox': [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                'img_size': [W, H],
            }],
            'source': 'share_annot',
        }
        records.append(record)

    print(f'[{split}] converted: {len(records)}  missing(images not found): {missing}')
    return records


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    train_records = convert_split('train')
    val_records = convert_split('val')

    # 写 train.json / val.json (覆盖旧文件)
    train_out = os.path.join(DATA_DIR, 'train.json')
    val_out = os.path.join(DATA_DIR, 'val.json')
    with open(train_out, 'w', encoding='utf-8') as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)
    with open(val_out, 'w', encoding='utf-8') as f:
        json.dump(val_records, f, ensure_ascii=False, indent=2)

    total = len(train_records) + len(val_records)
    print()
    print(f'Written: {train_out} ({len(train_records)} records, '
          f'{os.path.getsize(train_out)} bytes)')
    print(f'Written: {val_out} ({len(val_records)} records, '
          f'{os.path.getsize(val_out)} bytes)')
    print(f'Total: {total}  |  train {len(train_records)/total*100:.1f}%  '
          f'val {len(val_records)/total*100:.1f}%')

    # 抽样验证
    print()
    print('=== sample train[0] ===')
    print(json.dumps(train_records[0], ensure_ascii=False, indent=2))
    print()
    print('=== sample val[0] ===')
    print(json.dumps(val_records[0], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
