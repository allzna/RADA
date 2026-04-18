import os
import subprocess
import sys

# ================= 配置区域 =================
# 1. 设置数据根目录 (请根据您的实际路径修改)
#    注意：Windows路径建议使用正斜杠 '/' 或双反斜杠 '\\'
ROOT_PATH = "D:\Desktop\研究生学习\代码\数据集\HHAR-P\HHAR_SA"

# 2. 设置源域和目标域 ID
#    HHAR_P 的 Domain ID 通常为 '0', '1', '2' ... '8'
ID_SRC = "0"   # 源域
ID_TRG = "2"   # 目标域

# 3. 其他参数
DATA_NAME = "HHAR_P"
DATA_ORIGIN = "ACON"  # 告诉代码这是 ACON 格式的数据
GPU_ID = "0"
# ===========================================

def run_experiment():
    # 检查 run.py 是否存在
    if not os.path.exists("run.py"):
        print("错误：未找到 run.py 文件。请将此脚本放在 CrossAD 项目根目录下。")
        return

    # 构建命令行参数
    # 对应原 Shell 脚本：python -u run.py ...
    cmd = [
        sys.executable, "-u", "run.py",
        "--mode", "train",
        "--configs_path", "./configs/",
        "--save_path", "./test_results/",
        "--root_path", ROOT_PATH,
        "--data", DATA_NAME,
        "--data_origin", DATA_ORIGIN,
        "--id_src", ID_SRC,
        "--id_trg", ID_TRG,
        "--gpu", GPU_ID
    ]

    print(f"正在启动 {DATA_NAME} 实验 (Source: {ID_SRC} -> Target: {ID_TRG})...")
    print("执行命令:", " ".join(cmd))
    print("-" * 50)

    try:
        # 调用命令行执行
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[错误] 脚本运行失败，退出代码: {e.returncode}")
    except KeyboardInterrupt:
        print("\n[用户终止] 实验已手动停止。")

if __name__ == "__main__":
    run_experiment()