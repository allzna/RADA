import os
import subprocess
import json

# ⚠️ 保持与你主脚本一致
DATASET_ROOT = r"../datasets"
CACHE_FILE = "./test_results/all_results_cache.json"


def patch_run_zero_screening():
    if not os.path.exists(CACHE_FILE):
        print("⚠️ 找不到缓存文件！请确认路径。")
        return

    with open(CACHE_FILE, 'r') as f:
        all_results = json.load(f)

    for ds in ['MSL', 'SMAP', 'SMD']:
        if ds not in all_results: continue

        tasks = all_results[ds].get('completed_tasks', [])
        for i, task_id in enumerate(tasks):
            # 正常跑完 8 个参数是长度 8，如果长度已经是 9，说明补跑过了
            if len(all_results[ds]['x_ratios'][i]) >= 9:
                print(f"⏭️ 任务 {task_id} 已包含 -1.0 的结果，跳过。")
                continue

            source, target = task_id.split('_to_')
            print(f"\n🚀 [补跑 0% 端点] 任务: {ds} | [{source}] -> [{target}] @ mult = -1.0")

            # 调用 run.py，强行传入 -1.0 激活后门
            cmd = [
                "python", "run.py",
                "--mode", "train",
                "--root_path", DATASET_ROOT,
                "--data", ds,
                "--id_src", source,
                "--id_trg", target,
                "--data_origin", "DADA",
                "--gpu", "0",
                "--iqr_mult", "-1.0"
            ]

            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            result_dir = f"./test_results/{ds}/results_{task_id}"
            f1_score = 0.0

            try:
                # 读取跑出来的最新 F1 分数
                json_path = os.path.join(result_dir, "final_metrics.json")
                with open(json_path, 'r') as f:
                    metrics = json.load(f)
                    f1_score = float(metrics.get('best_f1_F1_best', metrics.get('best_f1_f1', 0.0)))
            except Exception as e:
                print(" (❌ JSON解析失败)", end="")

            print(f" [✅ 补跑完成 F1: {f1_score:.4f}, Filter: 0.00%]")

            # 🌟 直接向现有的列表末尾追加绝对 0.0 的横坐标，以及新跑出的 F1
            all_results[ds]['x_ratios'][i].append(0.0)
            all_results[ds]['y_f1s'][i].append(f1_score)

            # 实时保存，防止意外中断
            with open(CACHE_FILE, "w") as f:
                json.dump(all_results, f, indent=4)

    print("\n🎉 所有 0% 端点补跑完成！现在你可以直接运行 特色实验4-2改.py 画图了！")


if __name__ == '__main__':
    patch_run_zero_screening()