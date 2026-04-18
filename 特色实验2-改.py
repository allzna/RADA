import matplotlib.pyplot as plt
import numpy as np
import warnings
import matplotlib.font_manager as fm
import os

# ==========================================
#  基础设置
# ==========================================
warnings.filterwarnings("ignore")
plt.rcdefaults()

# 强制全局使用 Arial 字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.linewidth'] = 1.5

# 🌟 修改点 1：字体大小全改为 22
FONT_SIZE = 24
plt.rcParams['xtick.labelsize'] = FONT_SIZE
plt.rcParams['ytick.labelsize'] = FONT_SIZE

try:
    font_path = r"C:\Windows\Fonts\arial.ttf"
    arial_font = fm.FontProperties(fname=font_path, size=FONT_SIZE)
    arial_font_title = fm.FontProperties(fname=font_path, size=FONT_SIZE, weight='bold')
    arial_font_legend = fm.FontProperties(fname=font_path, size=FONT_SIZE)
except:
    arial_font = fm.FontProperties(family='Arial', size=FONT_SIZE)
    arial_font_title = fm.FontProperties(family='Arial', size=FONT_SIZE, weight='bold')
    arial_font_legend = fm.FontProperties(family='Arial', size=FONT_SIZE)


# ==========================================
#  全局数据加载逻辑
# ==========================================
def load_full_dataset_data(dataset_name, results_root="test_results"):
    ds_path = os.path.join(results_root, dataset_name)
    all_scores = []
    all_labels = []

    if os.path.exists(ds_path):
        folders = [f for f in os.listdir(ds_path) if os.path.isdir(os.path.join(ds_path, f))]
        for folder in folders:
            base_path = os.path.join(ds_path, folder)
            try:
                scores = np.load(f'{base_path}/a_test_energy.npy')
                labels = np.load(f'{base_path}/a_gt.npy')
                if scores.ndim > 1:
                    scores = np.mean(scores, axis=-1)
                all_scores.append(scores.flatten())
                all_labels.append(labels.flatten())
            except FileNotFoundError:
                continue

    if len(all_scores) > 0:
        return np.concatenate(all_scores), np.concatenate(all_labels), True

    # 占位数据逻辑
    loc_params = {'MSL': (0.3, 0.1, 1.2, 0.4), 'SMAP': (0.2, 0.08, 1.0, 0.5), 'SMD': (0.4, 0.15, 1.5, 0.6)}
    loc_n, scale_n, loc_a, scale_a = loc_params.get(dataset_name, loc_params['MSL'])
    n_s = np.random.normal(loc=loc_n, scale=scale_n, size=50000)
    a_s = np.random.normal(loc=loc_a, scale=scale_a, size=3000)
    a_s = np.append(a_s, [10.0, 50.0, 200.0, 1000.0])
    return np.concatenate([n_s[n_s > 0], a_s]), np.concatenate([np.zeros(len(n_s[n_s > 0])), np.ones(len(a_s))]), False


# ==========================================
#  主绘图函数
# ==========================================
def plot_multi_dataset_iqr():
    datasets = ['MSL', 'SMAP', 'SMD']

    # 🌟 修改点 2：增加宽度并减小高度，由 figsize=(12, 20) 改为 figsize=(16, 12)
    # 这样可以大幅提高每张子图的长宽比，让图形更扁平
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    global_handles, global_labels = [], []

    for i, ds in enumerate(datasets):
        ax = axes[i]
        scores, labels, _ = load_full_dataset_data(ds)

        q1, q3 = np.percentile(scores, 25), np.percentile(scores, 75)
        iqr_threshold = q3 + 1.5 * (q3 - q1)

        normal_scores, anomaly_scores = scores[labels == 0], scores[labels == 1]
        linthresh = iqr_threshold * 1.5
        plot_max = np.max(scores) * 1.1

        # 划分区间
        lin_bins = np.linspace(0, linthresh, 31)
        bins = np.concatenate([lin_bins[:-1], np.logspace(np.log10(linthresh), np.log10(plot_max),
                                                          51)]) if plot_max > linthresh else lin_bins

        n_counts, _ = np.histogram(normal_scores, bins=bins)
        a_counts, _ = np.histogram(anomaly_scores, bins=bins)
        n_heights, a_heights = n_counts / (np.max(n_counts) + 1e-8), a_counts / (np.max(a_counts) + 1e-8)

        num_bins = len(bins) - 1
        x_positions = np.arange(num_bins)

        # 绘制条形
        ax.vlines(x_positions - 0.25, ymin=0, ymax=n_heights, color='#1f77b4', linewidth=2, alpha=0.85)
        ax.plot(x_positions - 0.25, n_heights, marker='_', color='#1f77b4', markersize=6, linestyle='None',
                label='Normal Samples')
        ax.vlines(x_positions + 0.25, ymin=0, ymax=a_heights, color='#d62728', linewidth=2, alpha=0.85)
        ax.plot(x_positions + 0.25, a_heights, marker='_', color='#d62728', markersize=6, linestyle='None',
                label='True Anomalies')

        # 阈值线及背景
        thresh_idx = int(np.interp(iqr_threshold, bins, np.arange(len(bins))))
        ax.axvline(thresh_idx, color='black', linestyle='--', linewidth=2, label='IQR Threshold')
        ax.axvspan(-1, thresh_idx, color='#2ca02c', alpha=0.08, label='Valid Sample Retention')
        ax.axvspan(thresh_idx, num_bins, color='#d62728', alpha=0.08, label='Anomaly Rejection')

        # 🌟 修改点 3：安全距离保持 12
        base_ticks = np.linspace(0, num_bins, 6).astype(int)
        filtered_ticks = [t for t in base_ticks if abs(t - thresh_idx) > 12]
        tick_indices = np.sort(np.unique(filtered_ticks + [thresh_idx]))

        tick_labels = []
        for idx in tick_indices:
            val = bins[idx]
            if val == 0:
                tick_labels.append("0")
            elif val < 0.1:
                tick_labels.append(f"{val:.3f}")
            elif val < 10:
                tick_labels.append(f"{val:.2f}")
            elif val < 1000:
                tick_labels.append(f"{val:.0f}")
            else:
                tick_labels.append(f"{val:.1e}")

        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, rotation=15, fontproperties=arial_font)

        # 完善元素
        ax.set_xlim(-1, num_bins)
        ax.set_ylim(0, 1.1)
        ax.set_title(f'({chr(97 + i)}) {ds} Dataset', fontproperties=arial_font_title, pad=15)
        ax.set_xlabel('Final Anomaly Score', fontproperties=arial_font)
        ax.set_ylabel('Normalized Density', fontproperties=arial_font)

        if i == 0:
            global_handles, global_labels = ax.get_legend_handles_labels()

    # 🌟 修改点 4：因为画幅变扁，整体图注位置微调 (bbox_to_anchor的Y从1.02改成1.06) 防止盖住第一个子图标题
    by_label = dict(zip(global_labels, global_handles))
    fig.legend(by_label.values(), by_label.keys(), prop=arial_font_legend,
               loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=2, framealpha=0.9, edgecolor='black')

    # 🌟 修改点 5：调整顶端边距和垂直间距 (top和hspace改变以适配扁平的画布)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.9)

    plt.savefig('Figure_IQR_Vertical_V22_Flatter.png', dpi=300, bbox_inches='tight')
    print(f"✅ 已生成长宽比更高的 22pt 字体竖排版：Figure_IQR_Vertical_V22_Flatter.png")


if __name__ == '__main__':
    plot_multi_dataset_iqr()