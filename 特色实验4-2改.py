import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings

# ==========================================
#  基础美学设置 (兼容服务器环境)
# ==========================================
warnings.filterwarnings("ignore")
plt.rcdefaults()
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

arial_font       = fm.FontProperties(family='sans-serif', size=20)
arial_font_title = fm.FontProperties(family='sans-serif', size=20, weight='bold')
arial_font_legend = fm.FontProperties(family='sans-serif', size=20)

CACHE_FILE = "./test_results/all_results_cache.json"

# 红线的固定网格点（与实验设计对齐）
TARGET_FILTER_RATIOS = [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100]
# 每个网格点的搜索半径（%）
GRID_TOL = 8.0
# 聚堆判断：中间点范围 < 此值则排除该 task
CLUSTER_THRESH = 20.0

color_left  = '#1f77b4'
color_right = '#C0504D'
color_star  = '#E3B448'


def is_clustered(x_list):
    mid = [v for v in x_list if 0 < v < 100]
    return bool(mid) and (max(mid) - min(mid)) < CLUSTER_THRESH


def compute_mean_on_grid(x_runs, y_runs):
    """
    在固定网格 TARGET_FILTER_RATIOS 上计算均值。
      1. 排除筛选率聚堆的 task（中间点范围 < CLUSTER_THRESH）
      2. 每个网格点只纳入在该点 ±GRID_TOL 内有真实数据的 task
      3. 每个 task 在该网格点只贡献一次：取最近邻真实点的 y 值（不插值）
    返回 (grid_x, mean_y, count_per_point)
    """
    grid_x = np.array(TARGET_FILTER_RATIOS, dtype=float)
    mean_y = np.full(len(grid_x), np.nan)
    count_per_point = np.zeros(len(grid_x), dtype=int)

    valid_pairs = [
        (np.array(x, dtype=float), np.array(y, dtype=float))
        for x, y in zip(x_runs, y_runs)
        if not is_clustered(x)
    ]

    for gi, gx in enumerate(grid_x):
        vals = []
        for x_s, y_s in valid_pairs:
            dists = np.abs(x_s - gx)
            if dists.min() <= GRID_TOL:
                # 用两侧真实点线性插值，得到恰好在 gx 处的估计值
                y_interp = np.interp(gx, x_s, y_s)
                vals.append(y_interp)
        if vals:
            mean_y[gi] = np.mean(vals)
            count_per_point[gi] = len(vals)

    return grid_x, mean_y, count_per_point


def plot_dual_axis_lines():
    if not os.path.exists(CACHE_FILE):
        print(f"⚠️ 找不到缓存文件 {CACHE_FILE}，请确认路径！")
        return

    print("📖 正在读取数据并绘制双Y轴折线图...")
    with open(CACHE_FILE, 'r') as fh:
        all_results = json.load(fh)

    fig, axes = plt.subplots(3, 1, figsize=(10, 16))
    datasets = ['MSL', 'SMAP', 'SMD']
    global_handles, global_labels = [], []

    for i, ds in enumerate(datasets):
        ax1 = axes[i]
        ax1.set_title(f'({chr(97 + i)}) Robustness on {ds}',
                      fontproperties=arial_font_title, pad=15)

        if ds not in all_results or len(all_results[ds].get('x_ratios', [])) == 0:
            continue

        x_runs = all_results[ds]['x_ratios']   # list of lists (各 task 点数可能不同)
        y_runs = all_results[ds]['y_f1s']

        all_x_flat = [v for row in x_runs for v in row]
        all_y_flat = [v for row in y_runs for v in row]

        # -------------------------------------------------------
        # 【修复】先统一插值到公共网格，再求均值，移出内层循环
        # -------------------------------------------------------
        grid_x, mean_y, count_per_point = compute_mean_on_grid(x_runs, y_runs)
        n_valid = sum(1 for x in x_runs if not is_clustered(x))
        n_clustered = len(x_runs) - n_valid
        print(f"  {ds}: 总tasks={len(x_runs)}, 聚堆排除={n_clustered}, 有效={n_valid}")
        print(f"  各格参与task数: {count_per_point.tolist()}")

        ax2 = ax1.twinx()
        ax1.grid(True, alpha=0.3, zorder=0)

        # --------------------------------------------------
        # 第一层 (ax1)：各 task 独立轨迹（灰蓝细线）
        # --------------------------------------------------
        line_indiv = None
        for j, (x_r, y_f) in enumerate(zip(x_runs, y_runs)):
            x_r = np.array(x_r, dtype=float)
            y_f = np.array(y_f, dtype=float)
            sort_idx = np.argsort(x_r)
            lbl = 'Task-specific Trajectories (Left Axis)' if j == 0 else ""
            line_indiv, = ax1.plot(x_r[sort_idx], y_f[sort_idx],
                                   color=color_left, alpha=0.35,
                                   linewidth=1.5, marker='o', markersize=3,
                                   zorder=2, label=lbl)

        # --------------------------------------------------
        # 第二层 (ax2)：网格对齐均值主线
        # --------------------------------------------------
        line_mean, = ax2.plot(grid_x, mean_y,
                              color=color_right, linewidth=4.0,
                              marker='D', markersize=9,
                              markerfacecolor='white', markeredgewidth=2.5,
                              zorder=5, label='Grid-Aligned Mean Trend (Right)')

        best_idx = np.argmax(mean_y)
        best_x, best_y = grid_x[best_idx], mean_y[best_idx]
        star_plot = ax2.scatter(best_x, best_y,
                                color=color_star, marker='*', s=500,
                                edgecolor='black', linewidth=1.5,
                                zorder=10, label='Global Optimum Trend')
        ax2.axvline(best_x, color='gray', linestyle='--', alpha=0.7,
                    linewidth=2, zorder=1)

        # --------------------------------------------------
        # 轴范围与标签
        # --------------------------------------------------
        ax1.set_xlabel('Proportion of Screened Anomalies (%)', fontproperties=arial_font)
        ax1.set_xlim(left=-2, right=max(all_x_flat) + 4)

        y_min, y_max = min(all_y_flat), max(all_y_flat)
        ax1.set_ylim(max(0.0, y_min - 0.1), min(1.0, y_max + 0.1))

        mean_min, mean_max = mean_y.min(), mean_y.max()
        y_span = mean_max - mean_min
        if y_span < 0.03:
            mid = (mean_max + mean_min) / 2.0
            ax2.set_ylim(mid - 0.02, mid + 0.02)
        else:
            margin = y_span * 0.35
            ax2.set_ylim(mean_min - margin, mean_max + margin)

        ax1.set_ylabel('Absolute F1 Score', color=color_left, fontproperties=arial_font)
        ax1.tick_params(axis='y', colors=color_left)
        ax2.set_ylabel('Mean F1 Trend', color=color_right, fontproperties=arial_font)
        ax2.tick_params(axis='y', colors=color_right)

        if i == 0 and line_indiv is not None:
            global_handles = [line_indiv, line_mean, star_plot]
            global_labels  = [
                'Task-specific Trajectories (Left Axis)',
                'Mean F1 Trajectory (Right Axis)',
                'Global Optimum Trend'
            ]

    # ==========================================
    #  全局图例
    # ==========================================
    fig.legend(global_handles, global_labels,
               prop=arial_font_legend,
               loc='upper center', bbox_to_anchor=(0.5, 1.03), # 稍微下调一点防止超出边界
               ncol=2, framealpha=0.9, edgecolor='black',
               handletextpad=0.5)

    plt.tight_layout()
    # top 调大一点给图例留空间，将 wspace 改为 hspace 控制上下间距
    plt.subplots_adjust(top=0.93, hspace=0.4)

    save_path = 'Figure_Hyperparameter_Robustness_Final_Aesthetic.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"\n✅ 图已生成：{save_path}")


if __name__ == '__main__':
    plot_dual_axis_lines()