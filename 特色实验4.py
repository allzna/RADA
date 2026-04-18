import os
import subprocess
import pandas as pd
import numpy as np
import json

# ==========================================
#  配置区域
# ==========================================
PATH_TO_CROSSAD = r'.'
DATASET_ROOT = r'../datasets'
CACHE_FILE = "./test_results/all_results_cache.json"

# 目标筛选率列表（均匀分布，单位 %）
# 0% = 不筛选任何异常，100% = 全部筛掉
TARGET_FILTER_RATIOS = [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100]

datasets_to_run = ['MSL', 'SMAP', 'SMD']


# ==========================================
#  二分查找：给定目标筛选率，反求 mult
# ==========================================
def find_mult_for_target_ratio(scores, labels, target_ratio, tol=1.0):
    """
    基于真实标签的筛选率定义：
        filter_ratio = 被过滤掉的真实异常数 / 总真实异常数 * 100
    用二分查找找到使筛选率最接近 target_ratio 的 mult 值。
    """
    total_anomalies = np.sum(labels == 1)
    if total_anomalies == 0:
        return 1.0  # 没有标注异常，返回默认值

    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1

    # 边界情况：目标筛选率为0，返回一个极大的 mult（几乎不筛）
    if target_ratio <= 0:
        return -1.0  # 用 -1 作为"不筛选"的标记

    # 边界情况：目标筛选率为100，返回0（阈值=q3，筛掉所有高分样本）
    if target_ratio >= 100:
        return 0.0

    base_threshold = q3 + 1.5 * iqr

    # mult=0 时阈值为0，所有样本都被筛掉；mult越大阈值越高，筛得越少
    # 所以二分的方向：ratio太小（筛得不够）→ 减小mult；ratio太大 → 增大mult
    lo, hi = 0.0, 200.0
    for _ in range(60):
        mid = (lo + hi) / 2
        threshold = mid * base_threshold
        filtered = np.sum((scores > threshold) & (labels == 1))
        ratio = (filtered / total_anomalies) * 100.0

        if abs(ratio - target_ratio) < tol:
            return mid
        elif ratio < target_ratio:
            # 筛得不够多，降低阈值（减小 mult）
            hi = mid
        else:
            # 筛得太多，提高阈值（增大 mult）
            lo = mid
    return mid


def compute_filter_ratio(scores, labels, mult):
    """用与模型一致的公式计算实际筛选率"""
    if mult < 0:
        return 0.0
    total_anomalies = np.sum(labels == 1)
    if total_anomalies == 0:
        return 0.0
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    base_threshold = q3 + 1.5 * iqr
    threshold = mult * base_threshold
    filtered = np.sum((scores > threshold) & (labels == 1))
    return (filtered / total_anomalies) * 100.0


# ==========================================
#  数据集任务配置
# ==========================================
def get_tasks_for_dataset(dataset_name):
    sources = []
    files = []

    if dataset_name == 'SMD':
        data_files_dir = os.path.join(DATASET_ROOT, 'SMD', 'test')
        sources = ['1-1', '2-3', '3-7', '1-5']
        if os.path.exists(data_files_dir):
            all_files = os.listdir(data_files_dir)
            files = [os.path.splitext(n)[0].replace('machine-', '') for n in all_files if
                     n.endswith(('.npy', '.csv', '.txt'))]

    elif dataset_name in ['MSL', 'SMAP']:
        label_csv_path = os.path.join(DATASET_ROOT, 'MSL_SMAP', 'labeled_anomalies.csv')
        data_files_dir = os.path.join(DATASET_ROOT, dataset_name, 'test')
        if dataset_name == 'MSL':
            sources = ['F-5', 'C-1', 'D-14', 'P-10']
        else:
            sources = ['A-7', 'P-2', 'E-8', 'D-7']

        if os.path.exists(label_csv_path):
            csv_reader = pd.read_csv(label_csv_path)
            data_info = csv_reader[csv_reader['spacecraft'] == dataset_name]
            space_files = np.asarray(data_info['chan_id']).tolist()
            if os.path.exists(data_files_dir):
                all_names = [n.split('.')[0] for n in os.listdir(data_files_dir) if n.endswith(('.npy', '.csv'))]
                files = [f for f in all_names if f in space_files]
            else:
                files = space_files

    files = sorted(list(set(files)))
    return sources, files


def save_cache(all_results):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as fh:
        json.dump(all_results, fh, indent=4)


# ==========================================
#  主流程
# ==========================================
def run_all_experiments():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as fh:
            all_results = json.load(fh)
    else:
        all_results = {}

    for ds in datasets_to_run:
        if ds not in all_results:
            all_results[ds] = {
                'completed_tasks': [],
                # 细粒度断点续跑：记录每个 task 已完成的 ratio
                # 结构：{ task_id: { 'ratios': [...], 'x': [...], 'f1': [...], ... } }
                'task_progress': {},
                'x_ratios': [],
                'y_f1s': [],
                'mults': [],
                'y_auprs': [],
                'y_aurocs': []
            }
        # 兼容旧缓存（无 task_progress 字段）
        if 'task_progress' not in all_results[ds]:
            all_results[ds]['task_progress'] = {}

        sources, files = get_tasks_for_dataset(ds)
        if not files:
            print(f"⚠️ 跳过 {ds}：找不到数据文件。")
            continue

        for src in sources:
            for trg in files:
                if src == trg:
                    continue

                task_id = f"{src}_to_{trg}"

                # 整个 task 已全部完成，跳过
                if task_id in all_results[ds]['completed_tasks']:
                    print(f"⏭️ 任务 [{task_id}] 已完成，跳过。")
                    continue

                print(f"\n==================================================")
                print(f"🚀 开始跨域任务: {ds} | [{src}] -> [{trg}]")
                print(f"==================================================")

                # 初始化该 task 的进度记录
                progress = all_results[ds]['task_progress'].setdefault(task_id, {
                    'completed_ratios': [],
                    'x_ratios': [], 'y_f1s': [], 'mults': [],
                    'y_auprs': [], 'y_aurocs': []
                })

                # --------------------------------------------------
                # Step 1: 用 mult=-1（不筛选）先跑一次，拿到 scores 和 labels
                # 用于后续二分查找
                # 【修复问题4】：若结果文件已存在则直接加载，不重复跑
                # --------------------------------------------------
                probe_result_dir = f"./test_results/{ds}/results_{task_id}_mult-1"
                probe_actual_dir = os.path.join(probe_result_dir, ds, f"results_{task_id}")
                probe_labels_path = os.path.join(probe_actual_dir, 'real_labels.npy')

                if os.path.exists(probe_labels_path):
                    print("  >>> [Step 1] 检测到预跑缓存，直接加载...")
                else:
                    print("  >>> [Step 1] 预跑 mult=-1，获取分数分布...")
                    probe_cmd = [
                        "python", os.path.join(PATH_TO_CROSSAD, "run.py"),
                        "--mode", "train",
                        "--root_path", DATASET_ROOT,
                        "--data", ds,
                        "--id_src", src,
                        "--id_trg", trg,
                        "--data_origin", "DADA",
                        "--gpu", "0",
                        "--iqr_mult", "-1",
                        "--save_path", probe_result_dir
                    ]
                    subprocess.run(probe_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                try:
                    rec = np.load(os.path.join(probe_actual_dir, 'real_L_rec.npy'))
                    unc = np.load(os.path.join(probe_actual_dir, 'real_L_unc.npy'))
                    labels = np.load(probe_labels_path)
                    probe_scores = rec + 1.0 * unc
                    print(f"  ✅ 预跑成功，共 {len(probe_scores)} 个样本，"
                          f"真实异常数: {int(np.sum(labels == 1))}")
                except Exception as e:
                    print(f"  ❌ 预跑失败，跳过此任务: {e}")
                    continue

                # --------------------------------------------------
                # Step 2: 对每个目标筛选率，二分查找对应的 mult，然后跑实验
                # 【修复问题3】：按 ratio 粒度断点续跑
                # 【修复问题1】：ratio=0 直接复用 Step 1 结果，不重复跑
                # --------------------------------------------------
                for target_ratio in TARGET_FILTER_RATIOS:

                    # 该 ratio 已完成，跳过
                    if target_ratio in progress['completed_ratios']:
                        print(f"  ⏭️ ratio={target_ratio}% 已完成，跳过。")
                        continue

                    mult = find_mult_for_target_ratio(probe_scores, labels, target_ratio)
                    print(f"\n  >>> 目标筛选率={target_ratio}% → 对应 mult={mult:.4f}")

                    # 【修复问题1】ratio=0 (mult=-1) 复用 Step 1 的结果目录，无需重跑
                    if target_ratio == 0:
                        actual_result_dir = probe_actual_dir
                        print(f"  ♻️  ratio=0% 直接复用 Step 1 结果，跳过重复训练。")
                    else:
                        result_dir = f"./test_results/{ds}/results_{task_id}_ratio{target_ratio}"
                        cmd = [
                            "python", os.path.join(PATH_TO_CROSSAD, "run.py"),
                            "--mode", "train",
                            "--root_path", DATASET_ROOT,
                            "--data", ds,
                            "--id_src", src,
                            "--id_trg", trg,
                            "--data_origin", "DADA",
                            "--gpu", "0",
                            "--iqr_mult", str(mult),
                            "--save_path", result_dir
                        ]
                        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        actual_result_dir = os.path.join(result_dir, ds, f"results_{task_id}")

                    # 读取性能指标
                    f1_raw, auroc_raw, aupr_raw = 0.0, 0.0, 0.0
                    try:
                        json_path = os.path.join(actual_result_dir, "final_metrics.json")
                        with open(json_path, 'r') as fh:
                            metrics = json.load(fh)
                            f1_raw    = float(metrics.get('best_f1_F1_best', 0.0))
                            auroc_raw = float(metrics.get('auc_AUC_ROC',     0.0))
                            aupr_raw  = float(metrics.get('auc_AUC_PR',      0.0))
                    except Exception as e:
                        print(f"    ❌ JSON读取失败: {e}")

                    # 计算实际筛选率
                    actual_ratio = float(target_ratio)
                    try:
                        rec2    = np.load(os.path.join(actual_result_dir, 'real_L_rec.npy'))
                        unc2    = np.load(os.path.join(actual_result_dir, 'real_L_unc.npy'))
                        labels2 = np.load(os.path.join(actual_result_dir, 'real_labels.npy'))
                        scores2 = rec2 + 1.0 * unc2
                        actual_ratio = compute_filter_ratio(scores2, labels2, mult)
                    except Exception as e:
                        print(f"    ⚠️ 实际筛选率计算失败，使用目标值: {e}")

                    print(f"    [完成] 目标筛选率={target_ratio}% | 实际={actual_ratio:.1f}% | "
                          f"mult={mult:.4f} | F1={f1_raw:.4f} | AUROC={auroc_raw:.4f} | AUPR={aupr_raw:.4f}")

                    # 【修复问题3】每个 ratio 完成后立即写入进度并保存缓存
                    progress['completed_ratios'].append(target_ratio)
                    progress['x_ratios'].append(actual_ratio)
                    progress['y_f1s'].append(f1_raw)
                    progress['mults'].append(mult)
                    progress['y_auprs'].append(aupr_raw)
                    progress['y_aurocs'].append(auroc_raw)
                    save_cache(all_results)
                    print(f"    💾 ratio={target_ratio}% 进度已保存。")

                # 所有 ratio 完成后，将结果汇总到顶层列表，标记 task 完成
                all_results[ds]['completed_tasks'].append(task_id)
                all_results[ds]['x_ratios'].append(progress['x_ratios'])
                all_results[ds]['y_f1s'].append(progress['y_f1s'])
                all_results[ds]['mults'].append(progress['mults'])
                all_results[ds]['y_auprs'].append(progress['y_auprs'])
                all_results[ds]['y_aurocs'].append(progress['y_aurocs'])
                save_cache(all_results)
                print(f"\n  ✅ 任务 [{task_id}] 全部完成，缓存已更新。")

    print(f"\n🎉 所有实验完成，结果写入 {CACHE_FILE}")


if __name__ == '__main__':
    run_all_experiments()