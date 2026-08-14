"""
训练暂停/恢复/停止控制脚本。

通过在 output 目录创建/删除 flag 文件来控制训练进程,
训练器的 _check_flags() 会轮询这些文件并执行对应操作。

用法:
  python tools/pause_training.py pause     # 当前 epoch 结束后暂停
  python tools/pause_training.py resume    # 从暂停中恢复训练
  python tools/pause_training.py stop      # 当前 epoch 结束后停止训练
  python tools/pause_training.py status    # 查看训练状态 (flag + 最新指标)
  python tools/pause_training.py           # 不带参数 = status
"""
import os
import sys
import csv
import time
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# flag 文件名 (与 trainer.py 一致)
PAUSE_FLAG = 'pause.flag'
RESUME_FLAG = 'resume.flag'
STOP_FLAG = 'stop.flag'


def find_training_process():
    """检查是否有训练进程正在运行 (通过查找 python + train.py)。"""
    try:
        import subprocess
        if os.name == 'nt':
            result = subprocess.run(
                ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
                capture_output=True, text=True, timeout=10
            )
            return result.stdout.strip().split('\n')
        else:
            result = subprocess.run(
                ['pgrep', '-a', '-f', 'train.py'],
                capture_output=True, text=True, timeout=10
            )
            return result.stdout.strip().split('\n') if result.stdout.strip() else []
    except Exception:
        return None


def read_last_metrics(log_dir):
    """从 metrics_epoch.csv 和 metrics_batch.csv 读取最新指标。"""
    result = {'epoch_rows': [], 'batch_rows': []}

    epoch_csv = os.path.join(log_dir, 'metrics_epoch.csv')
    if os.path.exists(epoch_csv):
        with open(epoch_csv, 'r', encoding='utf-8') as f:
            result['epoch_rows'] = list(csv.DictReader(f))

    batch_csv = os.path.join(log_dir, 'metrics_batch.csv')
    if os.path.exists(batch_csv):
        with open(batch_csv, 'r', encoding='utf-8') as f:
            result['batch_rows'] = list(csv.DictReader(f))

    return result


def read_training_log_tail(log_dir, n=10):
    """读取 training.log 最后 n 行。"""
    log_file = os.path.join(log_dir, 'training.log')
    if not os.path.exists(log_file):
        return []
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [l.rstrip() for l in lines[-n:]]


def safe_float(val, default=None):
    try:
        return float(val) if val not in (None, '') else default
    except (ValueError, TypeError):
        return default


def cmd_pause(output_dir):
    """创建 pause.flag, 训练将在当前 epoch 结束后暂停。"""
    flag_path = os.path.join(output_dir, PAUSE_FLAG)

    if os.path.exists(os.path.join(output_dir, RESUME_FLAG)):
        os.remove(os.path.join(output_dir, RESUME_FLAG))

    with open(flag_path, 'w') as f:
        f.write(f'pause requested at {datetime.now().isoformat()}\n')

    print(f'  ✅ 已创建暂停标记: {flag_path}')
    print(f'  📌 训练将在当前 epoch 结束后自动暂停')
    print(f'  📌 暂停后可用以下命令恢复:')
    print(f'     python tools/pause_training.py resume')


def cmd_resume(output_dir):
    """创建 resume.flag, 从暂停状态恢复训练。"""
    flag_path = os.path.join(output_dir, RESUME_FLAG)

    if os.path.exists(os.path.join(output_dir, PAUSE_FLAG)):
        os.remove(os.path.join(output_dir, PAUSE_FLAG))

    with open(flag_path, 'w') as f:
        f.write(f'resume requested at {datetime.now().isoformat()}\n')

    print(f'  ✅ 已创建恢复标记: {flag_path}')
    print(f'  📌 训练进程将检测到该标记并恢复训练')


def cmd_stop(output_dir):
    """创建 stop.flag, 训练将在当前 epoch 结束后停止。"""
    flag_path = os.path.join(output_dir, STOP_FLAG)

    with open(flag_path, 'w') as f:
        f.write(f'stop requested at {datetime.now().isoformat()}\n')

    print(f'  ✅ 已创建停止标记: {flag_path}')
    print(f'  📌 训练将在当前 epoch 结束后自动停止并保存 checkpoint')


def cmd_status(output_dir, log_dir):
    """显示当前训练状态: flag 状态、最新指标、日志尾部。"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print(f'\n{"="*60}')
    print(f'  训练状态  |  {now}')
    print(f'{"="*60}')

    # --- Flag 状态 ---
    print(f'\n  📁 output_dir: {output_dir}')
    print(f'  📁 log_dir:    {log_dir}')

    flags = [
        (PAUSE_FLAG, '暂停', '🟡'),
        (RESUME_FLAG, '恢复', '🟢'),
        (STOP_FLAG, '停止', '🔴'),
    ]
    print(f'\n  🚩 Flag 状态:')
    any_flag = False
    for name, desc, icon in flags:
        path = os.path.join(output_dir, name)
        exists = os.path.exists(path)
        status = f'{icon} 存在' if exists else '⚪ 不存在'
        print(f'     {name:<16} {status}  ({desc})')
        if exists:
            any_flag = True

    if any_flag:
        print(f'\n  ⚠️  有 flag 文件存在, 训练器将在下次 _check_flags() 时处理')

    # --- 训练进程 ---
    procs = find_training_process()
    if procs is not None:
        train_procs = [p for p in procs if 'train.py' in p]
        if train_procs:
            print(f'\n  🔧 训练进程: 运行中 (PID 信息见下)')
            for p in train_procs[:3]:
                print(f'     {p[:100]}')
        else:
            print(f'\n  🔧 训练进程: 未检测到 train.py 进程')
    else:
        print(f'\n  🔧 训练进程: 无法检测')

    # --- 最新指标 ---
    metrics = read_last_metrics(log_dir)
    epoch_rows = metrics['epoch_rows']
    batch_rows = metrics['batch_rows']

    if epoch_rows:
        latest = epoch_rows[-1]
        ep = latest.get('epoch', '?')
        tl = safe_float(latest.get('train_loss'))
        vl = safe_float(latest.get('val_loss'))
        mp = safe_float(latest.get('mAP50'))
        lr = latest.get('lr', '?')

        print(f'\n  📊 最新 Epoch 指标 (共 {len(epoch_rows)} 个 epoch):')
        print(f'     Epoch:      {ep}')
        print(f'     Train Loss: {tl:.4f}' if tl is not None else '     Train Loss: -')
        print(f'     Val Loss:   {vl:.4f}' if vl is not None else '     Val Loss:   -')
        print(f'     mAP@50:     {mp:.4f}' if mp is not None else '     mAP@50:     -')
        print(f'     LR:         {lr}')

        # 最佳指标
        best_val = min((safe_float(r.get('val_loss'), float('inf')) for r in epoch_rows), default=float('inf'))
        best_map = max((safe_float(r.get('mAP50'), 0) for r in epoch_rows), default=0)
        print(f'\n  🏆 历史最佳: Val Loss={best_val:.4f}  mAP@50={best_map:.4f}')
    else:
        print(f'\n  📊 尚无 epoch 指标记录')

    if batch_rows:
        latest_batch = batch_rows[-1]
        bl = safe_float(latest_batch.get('loss'))
        print(f'\n  📝 最新 Batch: epoch={latest_batch.get("epoch", "?")}  '
              f'batch={latest_batch.get("batch_idx", "?")}  '
              f'loss={bl:.4f}' if bl is not None else
              f'\n  📝 最新 Batch: (loss 数据缺失)')
        print(f'     总 batch 数: {len(batch_rows)}')

    # --- 日志尾部 ---
    log_tail = read_training_log_tail(log_dir, n=8)
    if log_tail:
        print(f'\n  📜 training.log 最后 8 行:')
        for line in log_tail:
            print(f'     {line}')

    # --- 可用命令 ---
    print(f'\n  💡 可用命令:')
    print(f'     python tools/pause_training.py pause    # 暂停')
    print(f'     python tools/pause_training.py resume   # 恢复')
    print(f'     python tools/pause_training.py stop     # 停止')
    print(f'     python tools/pause_training.py status   # 查看状态')

    print(f'\n{"="*60}')


def main():
    parser = argparse.ArgumentParser(
        description='训练暂停/恢复/停止控制',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
命令说明:
  pause    创建 pause.flag, 训练在当前 epoch 结束后暂停
  resume   创建 resume.flag, 从暂停状态恢复训练
  stop     创建 stop.flag, 训练在当前 epoch 结束后停止
  status   查看当前训练状态 (默认)
        """
    )
    parser.add_argument('command', nargs='?', default='status',
                        choices=['pause', 'resume', 'stop', 'status'],
                        help='控制命令 (默认: status)')
    parser.add_argument('--output_dir', default=os.path.join(PROJECT_ROOT, 'output'),
                        help='训练输出目录 (默认: output)')
    parser.add_argument('--log_dir', default=os.path.join(PROJECT_ROOT, 'logs'),
                        help='日志目录 (默认: logs)')
    args = parser.parse_args()

    output_dir = args.output_dir
    log_dir = args.log_dir

    os.makedirs(output_dir, exist_ok=True)

    if args.command == 'pause':
        cmd_pause(output_dir)
    elif args.command == 'resume':
        cmd_resume(output_dir)
    elif args.command == 'stop':
        cmd_stop(output_dir)
    elif args.command == 'status':
        cmd_status(output_dir, log_dir)


if __name__ == '__main__':
    main()
