import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings

# ==========================================
#  基础美学设置
# ==========================================
warnings.filterwarnings("ignore")
plt.rcdefaults()
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

arial_font = fm.FontProperties(family='sans-serif', size=14)
arial_font_title = fm.FontProperties(family='sans-serif', size=16, weight='bold')
arial_font_legend = fm.FontProperties(family='sans-serif', size=13)

CACHE_FILE = "./test_results/all_results_cache.json"

def plot_anchored_mean():
    if not os.path.exists(CACHE_FILE):
        print(f"⚠️ 找不到缓存文件 {CACHE_FILE}")
        return

    print("📖 正在读取数据并应用 超参数锚定平均法 (Anchored Mean)...")
    with open(CACHE_FILE, 'r') as f:
        all_results = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    datasets = ['MSL', 'SMAP', 'SMD']

    global_handles, global_labels = [], []

    color_left = '#1f77b4'
    color_right = '#C0504D'
    color_star = '#E3B448'

    for i, ds in enumerate(datasets):
        ax1 = axes[i]

        if ds not in all_results or len(all_results[ds].get('x_ratios', [])) == 0:
            ax1.set_title(f'({chr(97 + i)}) Robustness on {ds}', fontproperties=arial_font_title, pad=15)
            continue

        x_matrix = np.array(all_results[ds]['x_ratios'])
        y_matrix = np.array(all_results[ds]['y_f1s'])

        if len(x_matrix.shape) != 2:
            continue

        all_x_flat = x_matrix.flatten()
        all_y_flat = y_matrix.flatten()

        ax2 = ax1.twinx()
        ax1.grid(True, alpha=0.3, zorder=0)

        # ==================================================
        #  第一层 (ax1)：绘制独立任务真实轨迹
        # ==================================================
        for j in range(len(x_matrix)):
            x_r = x_matrix[j]
            y_f = y_matrix[j]
            sort_idx = np.argsort(x_r)
            lbl = 'Task-specific Trajectories (Left)' if j == 0 else ""
            line_plot_indiv, = ax1.plot(x_r[sort_idx], y_f[sort_idx], color=color_left, alpha=0.35,
                                        linewidth=1.5, marker='o', markersize=3, zorder=2, label=lbl)

        # ==================================================
        #  第二层 (ax2)：🌟 核心修改 -> 超参数锚定均值法 🌟
        # ==================================================
        # 直接对 9 个超参数各自的 X 和 Y 独立求绝对均值，拒绝任何插值伪造！
        mean_x = np.mean(x_matrix, axis=0)
        mean_y = np.mean(y_matrix, axis=0)

        sort_idx_mean = np.argsort(mean_x)
        mean_x_sorted = mean_x[sort_idx_mean]
        mean_y_sorted = mean_y[sort_idx_mean]

        # 绘制基于超参数锚定的均值主线
        line_plot_mean, = ax2.plot(mean_x_sorted, mean_y_sorted, color=color_right, linewidth=4.0,
                                   marker='D', markersize=9, markerfacecolor='white', markeredgewidth=2.5,
                                   zorder=5, label='Anchored Mean Trend (Right)')

        # 寻找均值轨迹上的全局最高点打星
        best_idx = np.argmax(mean_y_sorted)
        best_x = mean_x_sorted[best_idx]
        best_y = mean_y_sorted[best_idx]

        star_plot = ax2.scatter(best_x, best_y, color=color_star, marker='*', s=500,
                                edgecolor='black', linewidth=1.5, zorder=10, label='Global Optimum Trend')
        ax2.axvline(best_x, color='gray', linestyle='--', alpha=0.7, linewidth=2, zorder=1)

        # ==================================================
        #  图表刻度与细节设定
        # ==================================================
        ax1.set_title(f'({chr(97 + i)}) Robustness on {ds}', fontproperties=arial_font_title, pad=15)
        ax1.set_xlabel('Proportion of Screened Anomalies (%)', fontproperties=arial_font)

        ax1.set_xlim(left=-2, right=np.max(all_x_flat) + 4)

        y_min, y_max = np.min(all_y_flat), np.max(all_y_flat)
        ax1.set_ylim(max(0.0, y_min - 0.1), min(1.0, y_max + 0.1))

        # 🌟 右 Y 轴：死平防重叠机制 (拯救 SMD 的视觉效果)
        mean_min, mean_max = np.min(mean_y_sorted), np.max(mean_y_sorted)
        y_span = mean_max - mean_min
        if y_span < 0.03:
            mid_point = (mean_max + mean_min) / 2.0
            ax2.set_ylim(mid_point - 0.015, mid_point + 0.015)
        else:
            margin = y_span * 0.35
            ax2.set_ylim(mean_min - margin, mean_max + margin)

        ax1.set_ylabel('Absolute F1 Score', color=color_left, fontproperties=arial_font)
        ax1.tick_params(axis='y', colors=color_left)

        ax2.set_ylabel('Mean F1 Trend', color=color_right, fontproperties=arial_font)
        ax2.tick_params(axis='y', colors=color_right)

        if i == 0:
            global_handles = [line_plot_indiv, line_plot_mean, star_plot]
            global_labels = ['Task-specific Trajectories (Left)', 'Anchored Mean Trend (Right)',
                             'Global Optimum Trend']

    fig.legend(global_handles, global_labels, prop=arial_font_legend,
               loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, framealpha=0.9, edgecolor='black',
               handletextpad=0.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, wspace=0.45)

    save_path = 'Figure_Hyperparameter_Robustness_Final_Anchored.png'
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    # plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')

    print(f"\n=====================================================")
    print(f"✅ 终极超参数锚定版已生成：{save_path}")
    print(f"=====================================================")

if __name__ == '__main__':
    plot_anchored_mean()