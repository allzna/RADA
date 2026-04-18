import os
import subprocess

DATASET = "SMAP"
SRC = "A-7"
TRG = "P-4"
ROOT_PATH = "../数据集"

print("==================================================")
print("🚀 [1/2] 正在训练【迁移前 (Source-Only)】模型...")
print("💡 使用'同域欺骗法' (Target = Source) 绕过底层 Bug")
print("==================================================")
cmd_source_only = [
    "python", "run.py", "--mode", "train", "--root_path", ROOT_PATH,
    "--data", DATASET, "--id_src", SRC, "--id_trg", SRC,
    "--data_origin", "DADA", "--gpu", "0"
]
subprocess.run(cmd_source_only)

print("\n==================================================")
print("🚀 [2/2] 正在训练【迁移后 (RADA)】模型 (开启 IQR=1.0 筛选)...")
print("==================================================")
cmd_rada = [
    "python", "run.py", "--mode", "train", "--root_path", ROOT_PATH,
    "--data", DATASET, "--id_src", SRC, "--id_trg", TRG,
    "--data_origin", "DADA", "--gpu", "0", "--iqr_mult", "1.0"
]
subprocess.run(cmd_rada)

print(f"\n✅ 两个模型均已训练完成！")
print(f"1. 迁移前模型: ./configs/{DATASET}/checkpoints_{SRC}_to_{SRC}/checkpoint.pth")
print(f"2. 迁移后模型: ./configs/{DATASET}/checkpoints_{SRC}_to_{TRG}/checkpoint.pth")
print("🎉 现在可以运行画图脚本了！")