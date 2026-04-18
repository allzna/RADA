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
plt.rcParams['grid.alpha'] = 0.4
plt.rcParams['axes.axisbelow'] = True

plt.rcParams['axes.linewidth'] = 1.5

# 设置 X 轴和 Y 轴刻度数字的字体大小
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# 设置 Arial FontProperties 用于特定文本部件
arial_font = fm.FontProperties(family='Arial', size=12)
arial_font_title = fm.FontProperties(family='Arial', size=14, weight='bold')

# ==========================================
#  辅助函数
# ==========================================
def robust_norm(data, p_max=99.0):
    """
    鲁棒归一化
    使用 99% 分位数作为最大值截断，防止极个别超级离群点把整个图压扁
    """
    max_val = np.percentile(data, p_max)
    min_val = np.min(data)

    # 截断离群点
    clipped_data = np.clip(data, min_val, max_val)
    # 归一化到 0-1
    return (clipped_data - min_val) / (max_val - min_val + 1e-8)

# ==========================================
#  主绘图函数
# ==========================================
def plot_joint_filtering_real():
    # ！！！！修改为你的实际路径！！！！
    data_path = "test_results/MSL/results_F-5_to_C-2"

    try:
        rec_raw = np.load(os.path.join(data_path, 'real_L_rec.npy')).flatten()
        unc_raw = np.load(os.path.join(data_path, 'real_L_unc.npy')).flatten()
        labels = np.load(os.path.join(data_path, 'real_labels.npy')).flatten()
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # 使用鲁棒归一化
    rec = robust_norm(rec_raw, p_max=99.0)
    unc = robust_norm(unc_raw, p_max=99.0)

    # 增大不确定性的权重
    lambda_weight = 1.5
    scores = rec + lambda_weight * unc

    q1, q3 = np.percentile(scores, 25), np.percentile(scores, 75)
    iqr_thresh = q3 + 1.5 * (q3 - q1)

    fig, ax = plt.subplots(figsize=(9, 7))

    # 画正常样本 (蓝色圆点)
    ax.scatter(rec[labels == 0], unc[labels == 0],
               c='#1f77b4', alpha=0.3, s=15, edgecolors='none',
               label='Normal samples')

    # 画异常样本 (红色叉号)
    ax.scatter(rec[labels == 1], unc[labels == 1],
               c='#d62728', marker='x', alpha=0.9, s=35, linewidth=1.5, zorder=5,
               label='Anomaly samples')

    # 绘制联合阈值边界
    x_line = np.linspace(0, iqr_thresh, 100)
    y_line = (iqr_thresh - x_line) / lambda_weight
    ax.plot(x_line, y_line, color='black', linestyle='--', linewidth=2, zorder=4,
            label='Joint threshold')

    # 单独依赖重构误差的截断线
    only_rec_thresh = np.percentile(rec, 95)
    ax.axvline(x=only_rec_thresh, color='green', linestyle=':', linewidth=2, zorder=3,
               label='Reconstruction error threshold')

    # # 高亮隐蔽异常区域
    # max_y = max(np.max(unc), np.max(y_line[y_line > 0])) * 1.05
    # ax.add_patch(plt.Rectangle((0, max(0, (iqr_thresh - only_rec_thresh) / lambda_weight)),
    #                            only_rec_thresh,
    #                            max_y,
    #                            fill=True, color='#ff7f0e', alpha=0.15, zorder=1,
    #                            label='Uncertainty Dominated Hidden Anomaly Region'))

    # 设置标题和坐标轴名称 (去除括号和公式)
    # ax.set_title('Joint Feature Space Distribution', fontproperties=arial_font_title, pad=15)
    ax.set_xlabel('Reconstruction Error Perspective', fontproperties=arial_font)
    ax.set_ylabel('Uncertainty Estimation Perspective', fontproperties=arial_font)
    # ax.set_ylabel('Uncertainty Estimation Perspective with Robust Norm', fontproperties=arial_font)

    # 限制显示范围
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    ax.legend(prop=arial_font, loc='upper left', framealpha=0.95)

    plt.tight_layout()
    plt.savefig('Figure_Joint_Filtering_RealData_Robust_English.png', dpi=300, bbox_inches='tight')
    print("图表已生成！请查看 Figure_Joint_Filtering_RealData_Robust_English.png")


if __name__ == '__main__':
    plot_joint_filtering_real()