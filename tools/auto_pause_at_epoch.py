"""
自动在指定 epoch 完成后暂停训练的监控脚本。

原理：
    训练进程的 _check_flags() 在每个 epoch 开始时检查 output/pause.flag 文件。
    本脚本监控 logs/training.log，当检测到目标 epoch 的 "Train Loss" 日志
    （表示该 epoch 已训练完成）时，创建 pause.flag 文件。
    训练进程会在下一个 epoch 开始时检测到标志并暂停，同时保存 latest_checkpoint.pth。

用法：
    python tools/auto_pause_at_epoch.py --epoch 14
    （epoch 从 0 开始计数，14 表示第 15 个 epoch）

恢复训练：
    方式1（训练进程仍在运行/暂停等待中）：
        python tools/pause_training.py resume
    方式2（训练进程已停止，需重新启动）：
        python tools/train.py --resume output/latest_checkpoint.pth
"""

import os
import re
import sys
import time
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_file = os.path.join(project_root, "logs", "training.log")
pause_flag_file = os.path.join(project_root, "output", "pause.flag")


def get_completed_epochs():
    """解析 training.log，返回已完成训练的 epoch 列表。"""
    if not os.path.exists(log_file):
        return []

    completed = []
    pattern = re.compile(r"Epoch (\d+): Train Loss =")

    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    ep = int(m.group(1))
                    if ep not in completed:
                        completed.append(ep)
    except Exception as e:
        print(f"[Monitor] Error reading log: {e}")

    return completed


def main():
    parser = argparse.ArgumentParser(description="Auto-pause training at a target epoch")
    parser.add_argument("--epoch", type=int, required=True,
                        help="Target epoch (0-indexed). Pause after this epoch completes training.")
    parser.add_argument("--interval", type=int, default=30,
                        help="Polling interval in seconds (default: 30)")
    args = parser.parse_args()

    target_epoch = args.epoch
    interval = args.interval

    print("=" * 60)
    print(f"[Auto-Pause Monitor]")
    print(f"  Target: pause after epoch {target_epoch} (the {target_epoch + 1}th epoch) completes")
    print(f"  Log file: {log_file}")
    print(f"  Pause flag: {pause_flag_file}")
    print(f"  Polling interval: {interval}s")
    print("=" * 60)

    # 如果标志已存在，说明已触发过
    if os.path.exists(pause_flag_file):
        print(f"[Monitor] Pause flag already exists. Training will pause at next epoch start.")
        return

    while True:
        completed = get_completed_epochs()

        if target_epoch in completed:
            print(f"\n[Monitor] Epoch {target_epoch} training completed!")
            print(f"[Monitor] Completed epochs so far: {sorted(completed)}")

            # 创建暂停标志
            os.makedirs(os.path.dirname(pause_flag_file), exist_ok=True)
            with open(pause_flag_file, "w") as f:
                f.write("pause")
            print(f"[Monitor] Pause flag created: {pause_flag_file}")
            print(f"[Monitor] Training will pause at epoch {target_epoch + 1} start.")
            print(f"[Monitor] To resume (if process still running): python tools/pause_training.py resume")
            print(f"[Monitor] To resume (if process stopped): python tools/train.py --resume output/latest_checkpoint.pth")
            return

        # 打印进度
        max_epoch = max(completed) if completed else -1
        remaining = target_epoch - max_epoch
        print(f"[Monitor] {time.strftime('%H:%M:%S')} | Completed up to epoch {max_epoch} "
              f"| Target: {target_epoch} | Remaining: {remaining} epochs", flush=True)

        time.sleep(interval)


if __name__ == "__main__":
    main()
