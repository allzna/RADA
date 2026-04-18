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

# 设置 X 轴和 Y 轴刻度数字的字体大小
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# 设置 Arial FontProperties 用于特定文本部件
arial_font = fm.FontProperties(family='Arial', size=14)
arial_font_title = fm.FontProperties(family='Arial', size=16, weight='bold')
arial_font_annotation = fm.FontProperties(family='Arial', size=14)


# ==========================================
#  数据加载 / 生成逻辑
# ==========================================
def load_or_generate_data(use_mock=True, base_path=""):
    if not use_mock:
        try:
            scores = np.load(f'{base_path}/a_test_energy.npy')
            labels = np.load(f'{base_path}/a_gt.npy')

            if scores.ndim > 1:
                scores = np.mean(scores, axis=-1)
            scores = scores.flatten()
            labels = labels.flatten()
            return scores, labels
        except Exception as e:
            print(f"Failed to load real data, using mock data. Error: {e}")

    # 模拟数据
    normal_scores = np.random.normal(loc=0.4, scale=0.15, size=8000)
    normal_scores = normal_scores[normal_scores > 0]
    anomaly_scores = np.random.normal(loc=1.2, scale=0.4, size=500)

    scores = np.concatenate([normal_scores, anomaly_scores])
    labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])

    return scores, labels


# ==========================================
#  主绘图函数
# ==========================================
def plot_iqr_filtering_experiment():
    base_path = "test_results/MSL/results_F-5_to_C-2"
    scores, labels = load_or_generate_data(use_mock=True, base_path=base_path)  # 记得改回 False 使用真实数据

    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    iqr_threshold = q3 + 1.5 * iqr

    print(f"Total samples: {len(scores)}")
    print(f"Q1: {q1:.3f}, Q3: {q3:.3f}, IQR: {iqr:.3f}")
    print(f"IQR Threshold: {iqr_threshold:.3f}")

    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    fig, ax = plt.subplots(figsize=(10, 6))

    plot_max = min(np.max(scores), iqr_threshold * 2.5)
    bins = np.linspace(0, plot_max, 100)
    bin_width = bins[1] - bins[0]

    # --- 核心修改：使用直方图统计密度，但绘制成误差棒/火柴棍形状 ---
    # 分别统计密度
    n_counts, _ = np.histogram(normal_scores, bins=bins, density=True)
    a_counts, _ = np.histogram(anomaly_scores, bins=bins, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # 设置并排显示的偏移量，避免两根线完全重叠
    offset = bin_width * 0.25

    # 绘制 Normal Samples
    x_n = bin_centers - offset
    ax.vlines(x_n, ymin=0, ymax=n_counts, color='#1f77b4', linewidth=1.5, alpha=0.85)
    ax.plot(x_n, n_counts, marker='_', color='#1f77b4', markersize=4, markeredgewidth=1.5, linestyle='None',
            label='Normal Samples')

    # 绘制 True Anomalies
    x_a = bin_centers + offset
    ax.vlines(x_a, ymin=0, ymax=a_counts, color='#d62728', linewidth=1.5, alpha=0.85)
    ax.plot(x_a, a_counts, marker='_', color='#d62728', markersize=4, markeredgewidth=1.5, linestyle='None',
            label='True Anomalies')

    # --- 绘制 IQR 关键线 ---
    ax.axvline(iqr_threshold, color='black', linestyle='--', linewidth=1.2,
               label=f'IQR Threshold')

    # --- 区域划分高亮 ---
    ax.axvspan(0, iqr_threshold, color='#2ca02c', alpha=0.08, label='Valid Sample Retention Area')
    ax.axvspan(iqr_threshold, plot_max, color='#d62728', alpha=0.08, label='Anomaly Rejection Area')

    # --- 完善图表元素 (文本转英文以适配 Arial) ---
    # ax.set_title('Analysis of IQR Reliability Filtering Mechanism', fontproperties=arial_font_title, pad=15)
    ax.set_xlabel('Final Anomaly Score', fontproperties=arial_font)
    ax.set_ylabel('Density', fontproperties=arial_font)
    ax.set_xlim(0, plot_max)
    ax.set_ylim(bottom=0)

    # 🌟 核心修改：只保留 X 轴的 0，隐藏 Y 轴的 0
    # 首先需要强制绘制一次以获取真实的 ticks
    fig.canvas.draw()

    # 获取 Y 轴的所有刻度
    yticks = ax.get_yticks()
    # 过滤掉数值为 0 的刻度
    ax.set_yticks(yticks[yticks != 0])

    ax.legend(prop=arial_font, loc='upper right', framealpha=0.95)

    plt.tight_layout()
    plt.savefig('Figure_IQR_Filtering_ErrorBar.png', dpi=300, bbox_inches='tight')
    print("特色实验2 (误差棒样式) 已生成：Figure_IQR_Filtering_ErrorBar.png")


if __name__ == '__main__':
    plot_iqr_filtering_experiment()