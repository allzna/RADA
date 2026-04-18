import os
import subprocess
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings

# ==========================================
#  实验参数与模式设置
# ==========================================
# ⚠️ 此时处于服务器真实运行模式，必须为 False！
USE_MOCK_PREVIEW = False

# ⚠️ 数据集根目录 (与 run_uda_loop.py 保持一致)
DATASET_ROOT = r"../datasets"
iqr_mults_to_test = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 100.0]

# 缓存文件路径
CACHE_FILE = "./test_results/all_results_cache.json"

# ==========================================
#  绘图基础设置 (服务器字体安全兼容版)
# ==========================================
warnings.filterwarnings("ignore")
plt.rcdefaults()
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.4
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

arial_font = fm.FontProperties(family='sans-serif', size=14)
arial_font_title = fm.FontProperties(family='sans-serif', size=16, weight='bold')
arial_font_legend = fm.FontProperties(family='sans-serif', size=14)


# ==========================================
#  🌟 复刻 run_uda_loop.py 的任务生成逻辑
# ==========================================
def get_tasks_for_dataset(dataset_name):
    tasks = []
    sources = []
    targets = []
    label_csv_path = ""
    data_files_dir = ""

    if dataset_name == 'SMD':
        data_files_dir = os.path.join(DATASET_ROOT, 'SMD', 'test')
        sources = ['1-1', '2-3', '3-7', '1-5']
    elif dataset_name == 'MSL':
        label_csv_path = os.path.join(DATASET_ROOT, 'MSL_SMAP', 'labeled_anomalies.csv')
        data_files_dir = os.path.join(DATASET_ROOT, 'MSL', 'test')
        sources = ['F-5','C-1', 'D-14', 'P-10']
    elif dataset_name == 'SMAP':
        label_csv_path = os.path.join(DATASET_ROOT, 'MSL_SMAP', 'labeled_anomalies.csv')
        data_files_dir = os.path.join(DATASET_ROOT, 'SMAP', 'test')
        sources = ['A-7', 'P-2', 'E-8', 'D-7']
    else:
        return []

    if dataset_name in ['MSL', 'SMAP'] and os.path.exists(label_csv_path):
        try:
            with open(label_csv_path, 'r') as file:
                csv_reader = pd.read_csv(file, delimiter=',')
            data_info = csv_reader[csv_reader['spacecraft'] == dataset_name]
            space_files = np.asarray(data_info['chan_id'])

            if os.path.exists(data_files_dir):
                all_files = os.listdir(data_files_dir)
                all_names = [name.split('.')[0] for name in all_files if name.endswith(('.npy', '.csv'))]
                targets = [file for file in all_names if file in space_files]
            else:
                targets = space_files.tolist()
            targets.sort()
        except Exception as e:
            pass

    elif dataset_name == 'SMD':
        if os.path.exists(data_files_dir):
            all_files = os.listdir(data_files_dir)
            valid_extensions = ('.npy', '.csv', '.txt')
            for name in all_files:
                if not name.endswith(valid_extensions):
                    continue
                base_name = os.path.splitext(name)[0]
                clean_name = base_name.replace('machine-', '') if base_name.startswith('machine-') else base_name
                targets.append(clean_name)
            targets = sorted(list(set(targets)))

    for src in sources:
        for trg in targets:
            if src != trg:
                tasks.append((src, trg))

    return tasks


# ==========================================
#  核心运行逻辑 (包含实时存档与断点续传)
# ==========================================
def run_real_experiments():
    # 确保结果文件夹存在
    os.makedirs("./test_results", exist_ok=True)

    # 1. 尝试加载历史进度 (断点续传)
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                all_results = json.load(f)
            print(f"🔄 成功加载历史存档，将继续未完成的任务...")
        except Exception as e:
            print("⚠️ 存档读取失败，将重新开始跑...")
            all_results = {}
    else:
        all_results = {}

    for ds in ['MSL', 'SMAP', 'SMD']:
        tasks = get_tasks_for_dataset(ds)
        if not tasks:
            continue

        # 初始化该数据集的字典结构
        if ds not in all_results:
            all_results[ds] = {'x_ratios': [], 'y_f1s': [], 'completed_tasks': []}

        print(f"\n=============================================")
        print(f"🌟 评估 {ds} 数据集，总任务数: {len(tasks)}")
        print(f"=============================================")

        for source, target in tasks:
            task_id = f"{source}_to_{target}"

            # 🌟 智能跳过：如果该任务已经保存在完成了的列表中，直接跳过！
            if task_id in all_results[ds].get('completed_tasks', []):
                print(f"⏭️ 任务 [{source}] -> [{target}] 已在存档中，跳过。")
                continue

            print(f"\n🚀 测试任务: [{source}] -> [{target}]")
            task_x, task_y = [], []

            for mult in iqr_mults_to_test:
                print(f"   >>> IQR Mult = {mult} ...", end="", flush=True)
                cmd = [
                    "python", "run.py",
                    "--mode", "train",
                    "--root_path", DATASET_ROOT,
                    "--data", ds,
                    "--id_src", source,
                    "--id_trg", target,
                    "--data_origin", "DADA",
                    "--gpu", "0",
                    "--iqr_mult", str(mult)
                ]

                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                result_dir = f"./test_results/{ds}/results_{source}_to_{target}"
                f1_score, filter_ratio = 0.0, 0.0

                try:
                    # 读取 F1
                    json_path = os.path.join(result_dir, "final_metrics.json")
                    with open(json_path, 'r') as f:
                        metrics = json.load(f)
                        f1_score = float(metrics.get('best_f1_F1_best', metrics.get('best_f1_f1', 0.0)))

                    # 读取分数并计算真实异常过滤比例
                    rec = np.load(os.path.join(result_dir, 'real_L_rec.npy'))
                    unc = np.load(os.path.join(result_dir, 'real_L_unc.npy'))
                    labels = np.load(os.path.join(result_dir, 'real_labels.npy'))

                    scores = rec + 1.0 * unc
                    q1 = np.percentile(scores, 25)
                    q3 = np.percentile(scores, 75)
                    iqr = q3 - q1
                    threshold = q3 + mult * iqr

                    total_anomalies = np.sum(labels == 1)
                    filtered_anomalies = np.sum((scores > threshold) & (labels == 1))

                    if total_anomalies > 0:
                        filter_ratio = (filtered_anomalies / total_anomalies) * 100.0
                except Exception as e:
                    print(" (❌ 解析失败)", end="")
                    pass

                print(f" [✅ F1: {f1_score:.4f}, Filter: {filter_ratio:.2f}%]")
                task_x.append(filter_ratio)
                task_y.append(f1_score)

            # 记录数据
            all_results[ds]['x_ratios'].append(task_x)
            all_results[ds]['y_f1s'].append(task_y)
            all_results[ds]['completed_tasks'].append(task_id)

            # 🌟 实时存档：每跑完一个 source->target，立刻把所有数据写入 JSON！
            with open(CACHE_FILE, "w") as f:
                json.dump(all_results, f, indent=4)
            print(f"   💾 进度已安全保存至 {CACHE_FILE}")

    return all_results


# ==========================================
#  恢复完整的 Mock 生成逻辑
# ==========================================
def generate_mock_results():
    print("\n[⚠️ 提示] 当前处于 USE_MOCK_PREVIEW 模式，正在生成排版预览图...")
    results = {}
    configs = {
        'MSL': {'base_f1': 0.88, 'peak_x': 15, 'drop': 0.15, 'noise': 0.04, 'x_shift': 0},
        'SMAP': {'base_f1': 0.85, 'peak_x': 25, 'drop': 0.20, 'noise': 0.05, 'x_shift': -5},
        'SMD': {'base_f1': 0.92, 'peak_x': 10, 'drop': 0.10, 'noise': 0.03, 'x_shift': 5}
    }

    for ds in ['MSL', 'SMAP', 'SMD']:
        cfg = configs[ds]
        ds_x_runs, ds_y_runs = [], []
        for task_idx in range(30):
            x_vals, y_vals = [], []
            for i, mult in enumerate(iqr_mults_to_test):
                # 让 mult=100 时逼近 0%
                base_ratio = 50 - (mult * 14) + cfg['x_shift']
                ratio = max(0.0, base_ratio + np.random.normal(0, 3))
                x_vals.append(ratio)

                dist_from_peak = abs(ratio - cfg['peak_x'])
                f1 = cfg['base_f1'] - (dist_from_peak * cfg['drop'] * 0.02) + np.random.normal(0, cfg['noise'])
                f1 = min(0.99, max(0.40, f1))
                y_vals.append(f1)

            sort_idx = np.argsort(x_vals)
            ds_x_runs.append(np.array(x_vals)[sort_idx])
            ds_y_runs.append(np.array(y_vals)[sort_idx])
        results[ds] = {'x_ratios': ds_x_runs, 'y_f1s': ds_y_runs}
    return results


# ==========================================
#  联合绘图逻辑 (恢复开关判断)
# ==========================================
def plot_robustness_experiment():
    # 🌟 修复：恢复开关判断！
    if USE_MOCK_PREVIEW:
        all_results = generate_mock_results()
    else:
        all_results = run_real_experiments()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.0))
    datasets = ['MSL', 'SMAP', 'SMD']

    theme_color = '#1f77b4'
    global_handles, global_labels = [], []

    for i, ds in enumerate(datasets):
        ax = axes[i]

        if ds not in all_results or len(all_results[ds].get('x_ratios', [])) == 0:
            continue

        x_runs = all_results[ds]['x_ratios']
        y_runs = all_results[ds]['y_f1s']

        all_x_flat = np.concatenate(x_runs)
        if len(all_x_flat) == 0: continue

        common_x = np.linspace(0, np.max(all_x_flat), 100)

        interp_y_runs = []
        for x_r, y_f in zip(x_runs, y_runs):
            x_r_unique = np.array(x_r) + np.random.rand(len(x_r)) * 1e-5
            sort_idx = np.argsort(x_r_unique)
            interp_y = np.interp(common_x, x_r_unique[sort_idx], np.array(y_f)[sort_idx])
            interp_y_runs.append(interp_y)

        interp_y_runs = np.array(interp_y_runs)
        mean_y = np.mean(interp_y_runs, axis=0)
        std_y = np.std(interp_y_runs, axis=0)

        # 1. 绘制方差阴影
        ax.fill_between(common_x, mean_y - std_y, mean_y + std_y, color=theme_color, alpha=0.15,
                        label='Variance across domains')

        # 2. 绘制均值主线
        ax.plot(common_x, mean_y, color=theme_color, linewidth=3.5, label='Average F1 Trend')

        # 3. 绘制密云散点
        for j in range(len(x_runs)):
            lbl = 'Task-specific observations' if j == 0 else ""
            ax.scatter(x_runs[j], y_runs[j], color=theme_color, alpha=0.25, s=30, edgecolors='none', zorder=2,
                       label=lbl)

        # 4. 高亮最佳参数点
        best_idx = np.argmax(mean_y)
        best_x = common_x[best_idx]
        best_y = mean_y[best_idx]

        ax.scatter(best_x, best_y, color='#d62728', marker='*', s=300, zorder=10, edgecolor='black', linewidth=1.0,
                   label='Optimal Operating Point')
        ax.axvline(best_x, color='gray', linestyle='--', alpha=0.7, linewidth=2, zorder=1,
                   label='Optimal Parameter Choice')

        # 图表元素
        ax.set_title(f'({chr(97 + i)}) Robustness on {ds}', fontproperties=arial_font_title, pad=15)
        ax.set_xlabel('Proportion of Screened Anomalies (%)', fontproperties=arial_font)

        if i == 0:
            ax.set_ylabel('F1 Score', fontproperties=arial_font)

        ax.set_xlim(left=0, right=np.max(all_x_flat) + 2)
        ax.set_ylim(max(0.4, np.min(mean_y - std_y) - 0.05), min(1.0, np.max(mean_y + std_y) + 0.05))

        if i == 0:
            global_handles, global_labels = ax.get_legend_handles_labels()

    # 绘制全局图例
    by_label = dict(zip(global_labels, global_handles))
    if by_label:
        ordered_labels = ['Average F1 Trend', 'Variance across domains', 'Task-specific observations',
                          'Optimal Operating Point', 'Optimal Parameter Choice']
        ordered_handles = [by_label[lbl] for lbl in ordered_labels]

        fig.legend(ordered_handles, ordered_labels, prop=arial_font_legend,
                   loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, framealpha=0.9, edgecolor='black',
                   handletextpad=0.5, columnspacing=1.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, wspace=0.18)

    # plt.savefig('Figure_Hyperparameter_Robustness_Overall.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('Figure_Hyperparameter_Robustness_Overall.png', dpi=600, bbox_inches='tight')

    if USE_MOCK_PREVIEW:
        print("\n=====================================================")
        print("✅ 预览完成：模拟排版图已生成，检查无误后请将 USE_MOCK_PREVIEW 设为 False 挂机。")
        print("=====================================================")
    else:
        print("\n=====================================================")
        print("✅ 真实跑库完成：参数鲁棒性全局图已生成！")
        print("=====================================================")


if __name__ == '__main__':
    plot_robustness_experiment()