"""
share_annot 数据集合并 + 按本项目 train/val 比例 (80%/20% 按图像组分层) 重新划分
并转换为与 data/train.json, data/val.json 一致的格式。

输入:
  share_annot_ac72f1d926bb2d23/share_annot_ac72f1d926bb2d23/
    train/approved.json  -> dict['data'] = {id: {bbox, query, visible, infrared, depth, width, height}}
    val/approved.json    -> 同上
  (可选) plan.json/qc.json 里的 QC 元数据也会一并保留

输出:
  data/share_merged_train.json  格式与 data/train.json 相同
  data/share_merged_val.json    格式与 data/val.json 相同
  data/share_merged_summary.txt 合并与划分统计
"""
import os
import sys
import json
import random
import re
from collections import defaultdict

# 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

SHARE_BASE = os.path.join(
    PROJECT_ROOT,
    'share_annot_ac72f1d926bb2d23',
    'share_annot_ac72f1d926bb2d23'
)
TEST_DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    '初赛数据集-基于大模型的多模态视觉理解与推理'
)

OUT_TRAIN = os.path.join(PROJECT_ROOT, 'data', 'share_merged_train.json')
OUT_VAL = os.path.join(PROJECT_ROOT, 'data', 'share_merged_val.json')
OUT_SUMMARY = os.path.join(PROJECT_ROOT, 'data', 'share_merged_summary.txt')

# 匹配本项目原始划分比例 (7620:1935 = 79.75%:20.25%)
# 由于 share_annot 是 400 个图像组, 以 80:20 图像组划分即可达到近似比例
TARGET_TRAIN_RATIO = 7620 / 9555  # 约 0.7975
TARGET_VAL_RATIO = 1 - TARGET_TRAIN_RATIO  # 约 0.2025

RANDOM_SEED = 20240809  # 固定种子，保证可复现


def load_approved(path):
    """从 approved.json 加载数据列表。返回 list[(sample_id, item_dict)]。"""
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    data = d.get('data', {})
    return list(data.items())


def norm_query_id(sample_id, index_in_group):
    """
    share_annot 的 sample_id 是 `{group_id}_{seq_no}`
    本项目 query_id 是 `{img_id}_{query_idx}` (6位零填充 img_id, query_idx 从 0)

    为避免与本项目现有 query_id 冲突, share_annot 的 img_id 用 group_id 零填充到6位
    query_idx 直接用该组内的顺序索引
    """
    m = re.match(r'^(\d+)_(\d+)$', sample_id)
    if not m:
        img_id_raw = sample_id
    else:
        img_id_raw = m.group(1)
    img_id = '%06d' % int(img_id_raw)
    return '%s_%d' % (img_id, index_in_group)


def to_project_format(sample_id, item, img_group_root):
    """
    share_annot item (含 norm cxcywh bbox, query, 相对路径) 转换为
    本项目 train.json 格式:

    {
        'query_id': '000023_0',
        'rgb': 'G:\\...\\visible\\000023.png',     # 映射到测试集的绝对路径 (若存在), 否则为原始 share_annot 路径
        'ir':  'G:\\...\\infrared\\000023.png',
        'depth': 'G:\\...\\depth\\000023.png',
        'text': 'query text',
        'annotations': [{
            'bbox': [x, y, w, h],   # 像素坐标 xywh, 与 val.json 格式一致
            'category_id': 0
        }]
    }
    """
    # bbox 转换: 归一化 cxcywh -> 像素 xywh
    cx, cy, bw, bh = item['bbox']
    W, H = item['width'], item['height']
    x = (cx - bw / 2) * W
    y = (cy - bh / 2) * H
    w = bw * W
    h = bh * H
    # 裁剪到有效范围 (share_annot 的 bbox 可能略超出边界)
    x = max(0.0, x)
    y = max(0.0, y)
    w = max(1.0, min(w, W - x))
    h = max(1.0, min(h, H - y))

    # 构造模态文件路径
    # share_annot 中的相对路径: 'Train/001/color/00000001.png', 'Train/001/infrared/00000001.png' 等
    # 尝试映射到初赛数据集(6位 img_id), 如果存在用初赛数据集绝对路径
    m = re.match(r'^(\d+)_(\d+)$', sample_id)
    img_id_6d = None
    if m:
        img_id_6d = '%06d' % int(m.group(1))

    def resolve(share_path, modality):
        # 优先测试集
        if img_id_6d is not None:
            test_mod = {'rgb': 'visible', 'ir': 'infrared', 'depth': 'depth'}[modality]
            candidate = os.path.join(TEST_DATA_ROOT, 'Images', test_mod, img_id_6d + '.png')
            if os.path.exists(candidate):
                return os.path.abspath(candidate).replace('\\', '/')
        # 否则构造 share_annot 绝对路径 (保留相对, 供后续定位)
        abs_p = os.path.join(PROJECT_ROOT, share_path)
        if os.path.exists(abs_p):
            return os.path.abspath(abs_p).replace('\\', '/')
        # 最终回退: 保留原始相对路径
        return share_path

    out = {
        'query_id': sample_id,  # 外层替换
        'rgb': resolve(item.get('visible', ''), 'rgb'),
        'ir': resolve(item.get('infrared', ''), 'ir'),
        'depth': resolve(item.get('depth', ''), 'depth'),
        'text': item.get('query', ''),
        'annotations': [{
            'bbox': [x, y, w, h],
            'category_id': 0,
            'bbox_norm_original': [cx, cy, bw, bh],  # 保留原始归一化坐标参考
            'img_size': [W, H],
        }],
        'source': 'share_annot',
    }
    return out


def group_by_image(samples_with_ids):
    """list[(id, item)] -> {img_group_id: list[(id, item)]}"""
    groups = defaultdict(list)
    for sid, it in samples_with_ids:
        m = re.match(r'^(\d+)_(\d+)$', sid)
        if m:
            grp = m.group(1)
        else:
            grp = sid.split('_')[0] if '_' in sid else sid
        groups[grp].append((sid, it))
    return groups


def split_groups(groups_dict, train_ratio=TARGET_TRAIN_RATIO, seed=RANDOM_SEED):
    """按图像组进行 80/20 划分，使组级比例接近目标比例"""
    rng = random.Random(seed)
    group_keys = sorted(groups_dict.keys())
    rng.shuffle(group_keys)
    n_total = len(group_keys)
    n_train = round(n_total * train_ratio)
    train_keys = set(group_keys[:n_train])
    val_keys = set(group_keys[n_train:])
    return train_keys, val_keys


def convert_all(groups_dict, keys_set):
    """将指定图像组集合中的样本转换为项目格式 list"""
    results = []
    for gkey in sorted(groups_dict.keys()):
        if gkey not in keys_set:
            continue
        items = groups_dict[gkey]
        for idx_in_group, (sid, item) in enumerate(items):
            out = to_project_format(sid, item, gkey)
            new_qid = norm_query_id(sid, idx_in_group)
            out['query_id'] = new_qid
            results.append(out)
    return results


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    print('=' * 70)
    print('share_annot 合并 + 重划分')
    print('=' * 70)
    print('Input train approved:', os.path.join(SHARE_BASE, 'train', 'approved.json'))
    print('Input val approved  :', os.path.join(SHARE_BASE, 'val', 'approved.json'))

    # 1. 加载 train + val approved, 合并
    tr_samples = load_approved(os.path.join(SHARE_BASE, 'train', 'approved.json'))
    va_samples = load_approved(os.path.join(SHARE_BASE, 'val', 'approved.json'))
    all_samples = tr_samples + va_samples
    print()
    print('share_annot train samples:', len(tr_samples))
    print('share_annot val   samples:', len(va_samples))
    print('Merged total      samples:', len(all_samples))

    # 2. 按图像组聚合
    groups = group_by_image(all_samples)
    n_groups = len(groups)
    group_sizes = [len(v) for v in groups.values()]
    print('Merged total image groups:', n_groups)
    print('Queries-per-group: min=%d max=%d avg=%.2f' % (
        min(group_sizes), max(group_sizes), sum(group_sizes) / len(group_sizes)))

    # 3. 按本项目比例 (≈80%:20%) 以图像组为单位重新划分
    train_keys, val_keys = split_groups(groups)
    n_train_groups = len(train_keys)
    n_val_groups = len(val_keys)
    print()
    print('Re-split target ratio: train=%.2f%%  val=%.2f%%' % (
        TARGET_TRAIN_RATIO * 100, TARGET_VAL_RATIO * 100))
    print('Image groups -> train: %d (%.2f%%)' % (
        n_train_groups, n_train_groups / n_groups * 100))
    print('Image groups -> val:   %d (%.2f%%)' % (
        n_val_groups, n_val_groups / n_groups * 100))

    # 4. 转换为项目格式
    train_out = convert_all(groups, train_keys)
    val_out = convert_all(groups, val_keys)
    print()
    print('Output train samples:', len(train_out))
    print('Output val   samples:', len(val_out))
    print('Sample ratio: train=%.2f%%  val=%.2f%%' % (
        len(train_out) / (len(train_out) + len(val_out)) * 100,
        len(val_out) / (len(train_out) + len(val_out)) * 100,
    ))

    # 5. 检查路径解析: 成功映射到初赛数据集的比例
    def analyze_paths(lst, name):
        ok_rgb = sum(1 for x in lst if TEST_DATA_ROOT.replace('\\', '/') in x['rgb'].replace('\\', '/'))
        ok_ir = sum(1 for x in lst if TEST_DATA_ROOT.replace('\\', '/') in x['ir'].replace('\\', '/'))
        ok_dp = sum(1 for x in lst if TEST_DATA_ROOT.replace('\\', '/') in x['depth'].replace('\\', '/'))
        print('%s paths in 初赛数据集 -> rgb:%d/%d ir:%d/%d depth:%d/%d' % (
            name, ok_rgb, len(lst), ok_ir, len(lst), ok_dp, len(lst)))
        # 图像 ID 重叠
        pids = set()
        for x in lst:
            m = re.match(r'^(\d+)_(\d+)$', x['query_id'])
            if m:
                pids.add(m.group(1))
        return pids

    train_pids = analyze_paths(train_out, 'train')
    val_pids = analyze_paths(val_out, 'val  ')
    overlap = train_pids & val_pids
    print('Overlap image ids between train/val:', len(overlap))
    assert len(overlap) == 0, '图像重叠! 按组划分失败'

    # 6. 写输出
    write_json(OUT_TRAIN, train_out)
    write_json(OUT_VAL, val_out)
    print()
    print('Write:', OUT_TRAIN, '(%d bytes)' % os.path.getsize(OUT_TRAIN))
    print('Write:', OUT_VAL, '(%d bytes)' % os.path.getsize(OUT_VAL))

    # 7. 写 summary
    lines = []
    lines.append('share_annot merge + re-split report')
    lines.append('=' * 60)
    lines.append('Total original:     %d (train_annot=%d + val_annot=%d)' % (
        len(all_samples), len(tr_samples), len(va_samples)))
    lines.append('Total image groups: %d' % n_groups)
    lines.append('Target ratio:       train=%.2f%%  val=%.2f%%' % (
        TARGET_TRAIN_RATIO * 100, TARGET_VAL_RATIO * 100))
    lines.append('')
    lines.append('After re-split:')
    lines.append('  train groups: %d / %d (%.2f%%)' % (
        n_train_groups, n_groups, n_train_groups / n_groups * 100))
    lines.append('  val   groups: %d / %d (%.2f%%)' % (
        n_val_groups, n_groups, n_val_groups / n_groups * 100))
    lines.append('  train queries: %d (%.2f%%)' % (
        len(train_out), len(train_out) / (len(train_out) + len(val_out)) * 100))
    lines.append('  val   queries: %d (%.2f%%)' % (
        len(val_out), len(val_out) / (len(train_out) + len(val_out)) * 100))
    lines.append('')
    lines.append('Output files:')
    lines.append('  ' + OUT_TRAIN)
    lines.append('  ' + OUT_VAL)
    lines.append('')
    lines.append('First 3 train items:')
    for r in train_out[:3]:
        lines.append('  query_id=%s text="%s" bbox=[%s]' % (
            r['query_id'], r['text'][:80],
            ', '.join('%.1f' % x for x in r['annotations'][0]['bbox'])))
    lines.append('')
    lines.append('First 3 val items:')
    for r in val_out[:3]:
        lines.append('  query_id=%s text="%s" bbox=[%s]' % (
            r['query_id'], r['text'][:80],
            ', '.join('%.1f' % x for x in r['annotations'][0]['bbox'])))
    with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('Write:', OUT_SUMMARY)


if __name__ == '__main__':
    main()
