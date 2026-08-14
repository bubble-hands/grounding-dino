"""
生成 50 条测试集裸跑结果的可视化对比报告

输出：
  - test_results/raw/comparison/  每条查询的对比图（原图+预测框 vs GT框）
  - test_results/raw/report.html  汇总 HTML 报告（含统计 + 所有对比图）
"""
import os
import sys
import json
import base64
import statistics
import torch
import cv2
import numpy as np
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

TEST_DATA = os.path.join(project_root, '初赛数据集-基于大模型的多模态视觉理解与推理')
VAL_JSON = os.path.join(project_root, 'data', 'val.json')
TRAIN_JSON = os.path.join(project_root, 'data', 'train.json')
SUMMARY_JSON = os.path.join(project_root, 'test_results', 'raw', 'summary.json')
COMP_DIR = os.path.join(project_root, 'test_results', 'raw', 'comparison')
REPORT_HTML = os.path.join(project_root, 'test_results', 'raw', 'report.html')


def norm_query_id(qid):
    """000023_001 -> 000023_1 (去掉 query idx 的前导零)"""
    parts = qid.rsplit('_', 1)
    if len(parts) == 2:
        return '%s_%d' % (parts[0], int(parts[1]))
    return qid


def load_gt_map(*json_paths):
    """从 train.json + val.json 构建 query_id -> gt_bbox(xyxy in pixel) 映射"""
    gt_map = {}
    for jp in json_paths:
        if not os.path.exists(jp):
            continue
        with open(jp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            qid = item.get('query_id', '')
            anns = item.get('annotations', [])
            if anns:
                bbox = anns[0]['bbox']  # [x, y, w, h] in pixel
                gt_map[qid] = {
                    'bbox_xyxy': [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                    'bbox_xywh': bbox,
                }
    return gt_map


def compute_iou(box1, box2):
    """两个 xyxy 框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def denorm_box(box_norm, orig_w, orig_h):
    """cxcywh(归一化) -> xyxy(像素)"""
    cx, cy, w, h = box_norm
    x1 = (cx - w / 2) * orig_w
    y1 = (cy - h / 2) * orig_h
    x2 = (cx + w / 2) * orig_w
    y2 = (cy + h / 2) * orig_h
    return [x1, y1, x2, y2]


def read_rgb_bgr(rgb_path):
    """读取 RGB 图为 BGR（cv2 格式），兼容中文路径"""
    pil = Image.open(rgb_path).convert('RGB')
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_box(img, box_xyxy, color, label, thickness=3):
    """在图上画框 + 标签"""
    result = img.copy()
    h, w = result.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(result, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
    cv2.putText(result, label, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return result


def make_comparison(rgb_img, pred_xyxy, gt_xyxy, iou, score, query_text, orig_size):
    """生成 3 列对比图：原图 | 预测(绿) | GT(红) + 叠加(黄)"""
    h, w = rgb_img.shape[:2]
    gap = 10
    title_h = 40
    info_h = 70
    panel_w = w
    panel_h = h
    canvas_w = panel_w * 4 + gap * 3
    canvas_h = panel_h + title_h + info_h
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 240

    # 标题
    titles = ['Original', 'Prediction (Green)', 'Ground Truth (Red)', 'Overlap']
    for i, t in enumerate(titles):
        x_off = i * (panel_w + gap)
        cv2.putText(canvas, t, (x_off + 10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)

    # 原图
    canvas[title_h:title_h + panel_h, 0:panel_w] = rgb_img

    # 预测框
    pred_img = draw_box(rgb_img, pred_xyxy, (0, 255, 0), 'Pred: %.3f' % score)
    canvas[title_h:title_h + panel_h, panel_w + gap:2 * panel_w + gap] = pred_img

    # GT 框
    gt_img = draw_box(rgb_img, gt_xyxy, (0, 0, 255), 'GT')
    canvas[title_h:title_h + panel_h, 2 * panel_w + 2 * gap:3 * panel_w + 2 * gap] = gt_img

    # 叠加
    overlap = draw_box(rgb_img, pred_xyxy, (0, 255, 0), 'Pred')
    overlap = draw_box(overlap, gt_xyxy, (0, 0, 255), 'GT')
    cv2.putText(overlap, 'IoU: %.3f' % iou, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    canvas[title_h:title_h + panel_h, 3 * panel_w + 3 * gap:4 * panel_w + 3 * gap] = overlap

    # 底部信息
    info_y = title_h + panel_h + 30
    text_line = query_text[:120] + ('...' if len(query_text) > 120 else '')
    cv2.putText(canvas, 'Q: ' + text_line, (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    iou_color = (0, 180, 0) if iou >= 0.5 else (0, 180, 255) if iou >= 0.3 else (0, 0, 200)
    cv2.putText(canvas, 'IoU=%.3f  Score=%.3f  %s' % (iou, score,
                'PASS' if iou >= 0.5 else 'WEAK' if iou >= 0.3 else 'FAIL'),
                (canvas_w - 500, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, iou_color, 2)
    return canvas


def main():
    os.makedirs(COMP_DIR, exist_ok=True)

    # 加载 GT (同时从 train.json 和 val.json 加载)
    gt_map = load_gt_map(TRAIN_JSON, VAL_JSON)
    print('GT 标注数量:', len(gt_map))

    # 加载预测结果
    with open(SUMMARY_JSON, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    results = summary['results']
    print('预测结果数量:', len(results))

    # 逐条处理
    records = []
    iou_list = []
    matched = 0
    for r in results:
        qid = r['query_id']
        qid_norm = norm_query_id(qid)
        gt = gt_map.get(qid_norm)

        # 读取 RGB 原图
        rgb_path = os.path.join(TEST_DATA, 'Images', 'visible', qid.split('_')[0] + '.png')
        if not os.path.exists(rgb_path):
            print('  跳过 %s (无 RGB)' % qid)
            continue
        rgb_img = read_rgb_bgr(rgb_path)
        h, w = rgb_img.shape[:2]

        # 预测框 -> 像素 xyxy
        pred_xyxy = denorm_box(r['pred_box_norm'], w, h)

        if gt is None:
            # 无 GT，只画预测
            iou = 0.0
            gt_xyxy = [0, 0, 0, 0]
            print('  %s: 无 GT 标注' % qid)
        else:
            gt_xyxy = gt['bbox_xyxy']
            iou = compute_iou(pred_xyxy, gt_xyxy)
            iou_list.append(iou)
            matched += 1

        # 生成对比图
        comp_img = make_comparison(rgb_img, pred_xyxy, gt_xyxy, iou,
                                   r['score'], r['query'], (w, h))
        comp_path = os.path.join(COMP_DIR, qid + '_cmp.jpg')
        cv2.imwrite(comp_path, comp_img, [cv2.IMWRITE_JPEG_QUALITY, 85])

        records.append({
            'query_id': qid,
            'query': r['query'],
            'score': r['score'],
            'iou': iou,
            'pred_box_norm': r['pred_box_norm'],
            'gt_xyxy': gt_xyxy,
            'pred_xyxy': pred_xyxy,
            'has_gt': gt is not None,
            'comp_path': os.path.relpath(comp_path, os.path.dirname(REPORT_HTML)),
            'img_w': w,
            'img_h': h,
        })
        print('  %s: IoU=%.3f score=%.3f' % (qid, iou, r['score']))

    # 统计
    n = len(records)
    n_with_gt = len(iou_list)
    avg_iou = statistics.mean(iou_list) if iou_list else 0
    max_iou = max(iou_list) if iou_list else 0
    min_iou = min(iou_list) if iou_list else 0
    median_iou = statistics.median(iou_list) if iou_list else 0
    pass_count = sum(1 for x in iou_list if x >= 0.5)
    weak_count = sum(1 for x in iou_list if 0.3 <= x < 0.5)
    fail_count = sum(1 for x in iou_list if x < 0.3)
    avg_score = statistics.mean([r['score'] for r in records])

    print('\n========== 统计 ==========')
    print('总样本: %d, 有GT: %d' % (n, n_with_gt))
    print('平均 IoU: %.4f, 中位数: %.4f' % (avg_iou, median_iou))
    print('最高 IoU: %.4f, 最低: %.4f' % (max_iou, min_iou))
    print('PASS(>=0.5): %d, WEAK(0.3-0.5): %d, FAIL(<0.3): %d' % (pass_count, weak_count, fail_count))
    print('平均 score: %.4f' % avg_score)

    # 生成 HTML 报告
    html = generate_html(records, {
        'total': n, 'matched': n_with_gt,
        'avg_iou': avg_iou, 'median_iou': median_iou,
        'max_iou': max_iou, 'min_iou': min_iou,
        'pass': pass_count, 'weak': weak_count, 'fail': fail_count,
        'avg_score': avg_score,
    })
    with open(REPORT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)
    print('\nHTML 报告已生成:', REPORT_HTML)
    print('对比图目录:', COMP_DIR)


def generate_html(records, stats):
    rows_html = ''
    for i, r in enumerate(records):
        iou_color = '#28a745' if r['iou'] >= 0.5 else '#ffc107' if r['iou'] >= 0.3 else '#dc3545'
        status = 'PASS' if r['iou'] >= 0.5 else 'WEAK' if r['iou'] >= 0.3 else 'FAIL'
        if not r['has_gt']:
            iou_color = '#6c757d'
            status = 'NO_GT'
        comp_abs = os.path.join(os.path.dirname(REPORT_HTML), r['comp_path'])
        with open(comp_abs, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
        query_short = r['query'][:100] + ('...' if len(r['query']) > 100 else '')
        rows_html += '''
        <div class="card">
            <div class="card-header">
                <span class="qid">#{idx} {qid}</span>
                <span class="status" style="background:{color}">{status}</span>
                <span class="metric">IoU: {iou:.3f}</span>
                <span class="metric">Score: {score:.3f}</span>
            </div>
            <div class="query-text">{query}</div>
            <img class="comp-img" src="data:image/jpeg;base64,{b64}" alt="{qid}">
        </div>
        '''.format(idx=i + 1, qid=r['query_id'], color=iou_color, status=status,
                   iou=r['iou'], score=r['score'], query=query_short,
                   b64=img_b64)

    pass_rate = (stats['pass'] / stats['matched'] * 100) if stats['matched'] else 0
    fail_rate = (stats['fail'] / stats['matched'] * 100) if stats['matched'] else 0
    weak_rate = (stats['weak'] / stats['matched'] * 100) if stats['matched'] else 0
    pass_dist = (stats['pass'] / stats['matched'] * 100) if stats['matched'] else 0
    weak_dist = (stats['weak'] / stats['matched'] * 100) if stats['matched'] else 0
    fail_dist = (stats['fail'] / stats['matched'] * 100) if stats['matched'] else 0

    template = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>GroundingDINO 裸跑测试集对比报告</title>
<style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: "Microsoft YaHei", "Segoe UI", sans-serif; background: #f5f5f5; color: #333; line-height: 1.6; }
    .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
    h1 { text-align: center; color: #1a1a2e; margin: 20px 0; font-size: 28px; }
    .subtitle { text-align: center; color: #666; margin-bottom: 30px; font-size: 14px; }
    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }
    .stat-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center; }
    .stat-card .label { font-size: 13px; color: #888; margin-bottom: 5px; }
    .stat-card .value { font-size: 28px; font-weight: bold; color: #1a1a2e; }
    .stat-card .value.green { color: #28a745; }
    .stat-card .value.red { color: #dc3545; }
    .stat-card .value.orange { color: #ffc107; }
    .stat-card .value.blue { color: #007bff; }
    .stat-card .sub { font-size: 12px; color: #aaa; margin-top: 4px; }
    .dist-bar { display: flex; height: 30px; border-radius: 4px; overflow: hidden; margin: 10px 0 30px; }
    .dist-bar div { display: flex; align-items: center; justify-content: center; color: white; font-size: 13px; font-weight: bold; }
    .dist-pass { background: #28a745; }
    .dist-weak { background: #ffc107; color: #333 !important; }
    .dist-fail { background: #dc3545; }
    .card { background: white; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); overflow: hidden; }
    .card-header { padding: 12px 20px; border-bottom: 1px solid #eee; display: flex; align-items: center; gap: 15px; flex-wrap: wrap; }
    .qid { font-weight: bold; color: #1a1a2e; font-size: 15px; }
    .status { padding: 3px 12px; border-radius: 12px; color: white; font-size: 12px; font-weight: bold; }
    .metric { font-size: 14px; color: #555; }
    .query-text { padding: 8px 20px; background: #fafafa; color: #444; font-size: 13px; font-style: italic; border-bottom: 1px solid #eee; }
    .comp-img { width: 100%; display: block; }
    .footer { text-align: center; color: #999; font-size: 12px; margin: 30px 0; }
    .legend { display: flex; gap: 20px; justify-content: center; margin-bottom: 20px; flex-wrap: wrap; }
    .legend-item { display: flex; align-items: center; gap: 6px; font-size: 13px; }
    .legend-color { width: 16px; height: 16px; border-radius: 3px; }
</style>
</head>
<body>
<div class="container">
    <h1>GroundingDINO 裸跑测试集可视化对比报告</h1>
    <div class="subtitle">模型类型: Raw (随机初始化, 无微调) | 测试样本: __TOTAL__ 条 | 有GT标注: __MATCHED__ 条</div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="label">平均 IoU</div>
            <div class="value blue">__AVG_IOU__</div>
            <div class="sub">中位数 __MEDIAN_IOU__</div>
        </div>
        <div class="stat-card">
            <div class="label">最高 IoU</div>
            <div class="value green">__MAX_IOU__</div>
        </div>
        <div class="stat-card">
            <div class="label">最低 IoU</div>
            <div class="value red">__MIN_IOU__</div>
        </div>
        <div class="stat-card">
            <div class="label">平均 Score</div>
            <div class="value">__AVG_SCORE__</div>
        </div>
        <div class="stat-card">
            <div class="label">PASS (IoU&gt;=0.5)</div>
            <div class="value green">__PASS__</div>
            <div class="sub">占比 __PASS_RATE__</div>
        </div>
        <div class="stat-card">
            <div class="label">FAIL (IoU&lt;0.3)</div>
            <div class="value red">__FAIL__</div>
            <div class="sub">占比 __FAIL_RATE__</div>
        </div>
    </div>

    <div class="legend">
        <div class="legend-item"><div class="legend-color" style="background:#28a745"></div>预测框 Pred (绿)</div>
        <div class="legend-item"><div class="legend-color" style="background:#dc3545"></div>真实标注 GT (红)</div>
        <div class="legend-item"><div class="legend-color" style="background:#ffc107"></div>IoU 指标</div>
    </div>

    <div class="dist-bar">
        <div class="dist-pass" style="width:__PASS_DIST__">PASS __PASS__</div>
        <div class="dist-weak" style="width:__WEAK_DIST__">WEAK __WEAK__</div>
        <div class="dist-fail" style="width:__FAIL_DIST__">FAIL __FAIL__</div>
    </div>

    <h2 style="margin: 20px 0; font-size: 20px; color: #1a1a2e;">逐条对比详情</h2>
    __ROWS__

    <div class="footer">
        每张对比图包含 4 列: 原图 | 预测框(绿) | 真实标注(红) | 叠加对比(含IoU)<br>
        报告生成自 test_results/raw/summary.json + data/train.json + data/val.json (合成GT标注)
    </div>
</div>
</body>
</html>
'''
    html = template
    html = html.replace('__TOTAL__', str(stats['total']))
    html = html.replace('__MATCHED__', str(stats['matched']))
    html = html.replace('__AVG_IOU__', '%.4f' % stats['avg_iou'])
    html = html.replace('__MEDIAN_IOU__', '%.4f' % stats['median_iou'])
    html = html.replace('__MAX_IOU__', '%.4f' % stats['max_iou'])
    html = html.replace('__MIN_IOU__', '%.4f' % stats['min_iou'])
    html = html.replace('__AVG_SCORE__', '%.4f' % stats['avg_score'])
    html = html.replace('__PASS__', str(stats['pass']))
    html = html.replace('__PASS_RATE__', '%.1f%%' % pass_rate)
    html = html.replace('__FAIL__', str(stats['fail']))
    html = html.replace('__FAIL_RATE__', '%.1f%%' % fail_rate)
    html = html.replace('__WEAK__', str(stats['weak']))
    html = html.replace('__PASS_DIST__', '%.1f%%' % pass_dist)
    html = html.replace('__WEAK_DIST__', '%.1f%%' % weak_dist)
    html = html.replace('__FAIL_DIST__', '%.1f%%' % fail_dist)
    html = html.replace('__ROWS__', rows_html)
    return html


if __name__ == '__main__':
    main()
