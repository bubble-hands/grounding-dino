"""
清理旧的训练输出文件，为重新训练做准备。
"""
import os
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    output_dir = os.path.join(PROJECT_ROOT, 'output')
    logs_dir = os.path.join(PROJECT_ROOT, 'logs')
    
    print(f"清理输出目录: {output_dir}")
    
    # 删除 output 目录中的所有文件
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            fpath = os.path.join(output_dir, f)
            try:
                if os.path.isfile(fpath):
                    os.remove(fpath)
                    print(f"  删除文件: {f}")
                elif os.path.isdir(fpath):
                    shutil.rmtree(fpath)
                    print(f"  删除目录: {f}")
            except Exception as e:
                print(f"  删除失败 {f}: {e}")
    else:
        os.makedirs(output_dir)
        print(f"  创建目录: {output_dir}")
    
    print(f"\n清理日志目录: {logs_dir}")
    
    # 清空 logs 目录
    if os.path.exists(logs_dir):
        for f in os.listdir(logs_dir):
            fpath = os.path.join(logs_dir, f)
            try:
                if os.path.isfile(fpath):
                    os.remove(fpath)
                    print(f"  删除文件: {f}")
            except Exception as e:
                print(f"  删除失败 {f}: {e}")
    else:
        os.makedirs(logs_dir)
        print(f"  创建目录: {logs_dir}")
    
    # 删除 flag 文件
    for flag in ['pause.flag', 'resume.flag', 'stop.flag']:
        flag_path = os.path.join(output_dir, flag)
        if os.path.exists(flag_path):
            os.remove(flag_path)
            print(f"\n删除 flag: {flag}")
    
    print("\n清理完成！")
    print("可以开始新的训练了。")

if __name__ == '__main__':
    main()