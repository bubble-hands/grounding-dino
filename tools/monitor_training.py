"""
训练实时监控脚本。

功能:
  1. 实时读取 logs/metrics_batch.csv 和 logs/metrics_epoch.csv
  2. 终端打印最新 batch loss、epoch 汇总 (train_loss / val_loss / mAP50 / lr)
  3. 可选生成 matplotlib 可视化图表 (loss 曲线 + mAP 曲线) 保存为 PNG

用法:
  # 实时监控 (每 5 秒刷新)
  python tools/monitor_training.py

  # 指定日志目录和刷新间隔
  python tools/monitor_training.py --log_dir logs --refresh 10

  # 生成图表 (保存到 log_dir/training_plots.png)
  python tools/monitor_training.py --plot

  # 单次输出 (不循环)
  python tools/monitor_training.py --once
"""
import os
import sys
import csv
import time
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_csv(path):
    """读取 CSV 文件, 返回 list[dict]。文件不存在时返回空列表。"""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def safe_float(val, default=None):
    try:
        return float(val) if val != '' else default
    except (ValueError, TypeError):
        return default


def print_summary(batch_rows, epoch_rows):
    """打印当前训练状态摘要。"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'\n{"="*60}')
    print(f'  训练监控  |  {now}')
    print(f'{"="*60}')

    # --- Epoch 汇总 ---
    if epoch_rows:
        print(f'\n  📊 Epoch 汇总 (共 {len(epoch_rows)} 个 epoch):')
        print(f'  {"Epoch":>6}  {"Train Loss":>12}  {"Val Loss":>12}  {"mAP@50":>10}  {"LR":>12}  {"耗时":>8}')
        print(f'  {"-"*6}  {"-"*12}  {"-"*12}  {"-"*10}  {"-"*12}  {"-"*8}')
        for r in epoch_rows[-10:]:  # 最近 10 个 epoch
            ep = r.get('epoch', '?')
            tl = safe_float(r.get('train_loss'))
            vl = safe_float(r.get('val_loss'))
            mp = safe_float(r.get('mAP50'))
            lr = r.get('lr', '?')
            el = safe_float(r.get('elapsed_s'))
            tl_s = f'{tl:.4f}' if tl is not None else '-'
            vl_s = f'{vl:.4f}' if vl is not None else '-'
            mp_s = f'{mp:.4f}' if mp is not None else '-'
            el_s = f'{el:.1f}s' if el is not None else '-'
            print(f'  {ep:>6}  {tl_s:>12}  {vl_s:>12}  {mp_s:>10}  {lr:>12}  {el_s:>8}')

        # 趋势分析
        if len(epoch_rows) >= 2:
            latest = epoch_rows[-1]
            prev = epoch_rows[-2]
            lt = safe_float(latest.get('train_loss'))
            pt = safe_float(prev.get('train_loss'))
            lv = safe_float(latest.get('val_loss'))
            pv = safe_float(prev.get('val_loss'))
            lm = safe_float(latest.get('mAP50'))
            pm = safe_float(prev.get('mAP50'))

            print(f'\n  📈 趋势 (对比上一 epoch):')
            if lt is not None and pt is not None:
                delta = lt - pt
                arrow = '↓' if delta < 0 else '↑'
                print(f'     Train Loss: {pt:.4f} -> {lt:.4f}  {arrow} {abs(delta):.4f}')
            if lv is not None and pv is not None:
                delta = lv - pv
                arrow = '↓ (好)' if delta < 0 else '↑ (过拟合?)'
                print(f'     Val Loss:   {pv:.4f} -> {lv:.4f}  {arrow}')
            if lm is not None and pm is not None:
                delta = lm - pm
                arrow = '↑ (好)' if delta > 0 else '↓'
                print(f'     mAP@50:     {pm:.4f} -> {lm:.4f}  {arrow} {abs(delta):.4f}')

            # 最佳 epoch
            best_map = max((safe_float(r.get('mAP50'), 0) for r in epoch_rows), default=0)
            best_val = min((safe_float(r.get('val_loss'), float('inf')) for r in epoch_rows), default=float('inf'))
            print(f'\n  🏆 最佳: Val Loss={best_val:.4f}  mAP@50={best_map:.4f}')
    else:
        print('\n  ⏳ 尚无 epoch 记录 (等待第一个 epoch 完成...)')

    # --- Batch 级别 ---
    if batch_rows:
        latest_batch = batch_rows[-1]
        ep = latest_batch.get('epoch', '?')
        bi = latest_batch.get('batch_idx', '?')
        bl = safe_float(latest_batch.get('loss'))
        bl_s = f'{bl:.4f}' if bl is not None else '-'

        # 最近 20 个 batch 的平均 loss
        recent_losses = [safe_float(r.get('loss')) for r in batch_rows[-20:]]
        recent_losses = [l for l in recent_losses if l is not None]
        avg_recent = sum(recent_losses) / len(recent_losses) if recent_losses else 0

        # 第一个 batch 的 loss (用于对比)
        first_loss = safe_float(batch_rows[0].get('loss'))

        print(f'\n  📝 最新 Batch: epoch={ep}  batch={bi}  loss={bl_s}')
        print(f'     最近 20 batch 平均 loss: {avg_recent:.4f}')
        if first_loss is not None and bl is not None:
            delta = bl - first_loss
            arrow = '↓' if delta < 0 else '↑'
            print(f'     对比首个 batch: {first_loss:.4f} -> {bl_s}  {arrow} {abs(delta):.4f}')

        # 损失分量
        ce = safe_float(latest_batch.get('loss_ce'))
        bb = safe_float(latest_batch.get('loss_bbox'))
        gi = safe_float(latest_batch.get('loss_giou'))
        if ce is not None or bb is not None or gi is not None:
            ce_s = f'{ce:.4f}' if ce is not None else '-'
            bb_s = f'{bb:.4f}' if bb is not None else '-'
            gi_s = f'{gi:.4f}' if gi is not None else '-'
            print(f'     损失分量: CE={ce_s}  BBox={bb_s}  GIoU={gi_s}')

    print(f'\n{"="*60}')


def generate_plots(epoch_rows, batch_rows, save_path):
    """生成 loss 和 mAP 可视化图表, 保存为 PNG。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  [WARN] matplotlib 未安装, 跳过图表生成。安装: pip install matplotlib')
        return False

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Training Metrics', fontsize=14, fontweight='bold')

    # --- 1. Epoch Loss 曲线 ---
    ax = axes[0, 0]
    if epoch_rows:
        eps = [safe_float(r.get('epoch'), 0) for r in epoch_rows]
        tl = [safe_float(r.get('val_loss')) for r in epoch_rows]  # will fix below
        train_losses = [safe_float(r.get('train_loss')) for r in epoch_rows]
        val_losses = [safe_float(r.get('val_loss')) for r in epoch_rows]
        ax.plot(eps, train_losses, 'b-o', label='Train Loss', markersize=4)
        val_pts = [(e, v) for e, v in zip(eps, val_losses) if v is not None]
        if val_pts:
            ax.plot([p[0] for p in val_pts], [p[1] for p in val_pts], 'r-s', label='Val Loss', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Epoch Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No epoch data yet', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Epoch Loss')

    # --- 2. mAP 曲线 ---
    ax = axes[0, 1]
    map_pts = [(safe_float(r.get('epoch'), 0), safe_float(r.get('mAP50')))
               for r in epoch_rows if safe_float(r.get('mAP50')) is not None]
    if map_pts:
        ax.plot([p[0] for p in map_pts], [p[1] for p in map_pts], 'g-D', label='mAP@50', markersize=5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP@50')
        ax.set_title('Validation mAP@50')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    else:
        ax.text(0.5, 0.5, 'No mAP data yet', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Validation mAP@50')

    # --- 3. Batch Loss 曲线 ---
    ax = axes[1, 0]
    if batch_rows:
        indices = list(range(len(batch_rows)))
        losses = [safe_float(r.get('loss')) for r in batch_rows]
        valid = [(i, l) for i, l in zip(indices, losses) if l is not None]
        if valid:
            ax.plot([p[0] for p in valid], [p[1] for p in valid], 'b-', alpha=0.5, linewidth=0.5)
            # 移动平均
            window = min(50, len(valid) // 5) if len(valid) > 50 else 1
            if window > 1:
                ma_x = [p[0] for p in valid][window - 1:]
                ma_y = []
                vals = [p[1] for p in valid]
                for i in range(window - 1, len(vals)):
                    ma_y.append(sum(vals[i - window + 1:i + 1]) / window)
                ax.plot(ma_x, ma_y, 'r-', linewidth=1.5, label=f'MA({window})')
                ax.legend()
        ax.set_xlabel('Batch Index')
        ax.set_ylabel('Loss')
        ax.set_title('Batch Training Loss')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No batch data yet', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Batch Training Loss')

    # --- 4. Loss 分量曲线 ---
    ax = axes[1, 1]
    if batch_rows:
        indices = list(range(len(batch_rows)))
        ce_vals = [safe_float(r.get('loss_ce')) for r in batch_rows]
        bb_vals = [safe_float(r.get('loss_bbox')) for r in batch_rows]
        gi_vals = [safe_float(r.get('loss_giou')) for r in batch_rows]
        for vals, label, color in [(ce_vals, 'CE', 'blue'), (bb_vals, 'BBox', 'orange'), (gi_vals, 'GIoU', 'green')]:
            valid = [(i, v) for i, v in zip(indices, vals) if v is not None]
            if valid:
                ax.plot([p[0] for p in valid], [p[1] for p in valid], color=color, alpha=0.6, linewidth=0.8, label=label)
        ax.set_xlabel('Batch Index')
        ax.set_ylabel('Loss Component')
        ax.set_title('Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No batch data yet', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Loss Components')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  📊 图表已保存: {save_path}')
    return True


def main():
    parser = argparse.ArgumentParser(description='训练实时监控')
    parser.add_argument('--log_dir', default=os.path.join(PROJECT_ROOT, 'logs'),
                        help='日志目录 (默认: logs)')
    parser.add_argument('--refresh', type=int, default=5,
                        help='刷新间隔秒数 (默认: 5)')
    parser.add_argument('--plot', action='store_true',
                        help='生成并保存可视化图表')
    parser.add_argument('--once', action='store_true',
                        help='只输出一次, 不循环监控')
    args = parser.parse_args()

    log_dir = args.log_dir
    batch_csv = os.path.join(log_dir, 'metrics_batch.csv')
    epoch_csv = os.path.join(log_dir, 'metrics_epoch.csv')
    plot_path = os.path.join(log_dir, 'training_plots.png')

    print(f'监控日志目录: {log_dir}')
    print(f'  batch 指标: {batch_csv}')
    print(f'  epoch 指标: {epoch_csv}')
    if args.plot:
        print(f'  图表输出:   {plot_path}')

    if args.once:
        batch_rows = read_csv(batch_csv)
        epoch_rows = read_csv(epoch_csv)
        print_summary(batch_rows, epoch_rows)
        if args.plot:
            generate_plots(epoch_rows, batch_rows, plot_path)
        return

    # 实时循环监控
    last_batch_count = 0
    last_epoch_count = 0
    refresh = max(1, args.refresh)

    while True:
        try:
            batch_rows = read_csv(batch_csv)
            epoch_rows = read_csv(epoch_csv)

            # 清屏
            os.system('cls' if os.name == 'nt' else 'clear')

            print_summary(batch_rows, epoch_rows)

            # 有新数据时生成图表
            if args.plot and (len(batch_rows) != last_batch_count or len(epoch_rows) != last_epoch_count):
                generate_plots(epoch_rows, batch_rows, plot_path)
                last_batch_count = len(batch_rows)
                last_epoch_count = len(epoch_rows)

            print(f'\n  (每 {refresh}s 刷新 | Ctrl+C 退出)')
            time.sleep(refresh)
        except KeyboardInterrupt:
            print('\n\n  监控已停止。')
            break
        except Exception as e:
            print(f'\n  [ERROR] {e}')
            time.sleep(refresh)


if __name__ == '__main__':
    main()
