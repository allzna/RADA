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

# 🌟 修改点 1：全局字体大小统一设为 22
FONT_SIZE = 22
plt.rcParams['xtick.labelsize'] = FONT_SIZE
plt.rcParams['ytick.labelsize'] = FONT_SIZE

# 设置 Arial FontProperties 用于特定文本部件
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
#  全局数据加载逻辑
# ==========================================
def load_full_dataset_data(dataset_name, results_root="test_results"):
    ds_path = os.path.join(results_root, dataset_name)
    all_rec, all_unc, all_labels = [], [], []

    if os.path.exists(ds_path):
        folders = [f for f in os.listdir(ds_path) if os.path.isdir(os.path.join(ds_path, f))]
        for folder in folders:
            base_path = os.path.join(ds_path, folder)
            try:
                # 加载联合特征空间的数据
                rec = np.load(f'{base_path}/real_L_rec.npy').flatten()
                unc = np.load(f'{base_path}/real_L_unc.npy').flatten()
                labels = np.load(f'{base_path}/real_labels.npy').flatten()

                all_rec.append(rec)
                all_unc.append(unc)
                all_labels.append(labels)
            except FileNotFoundError:
                continue

    if len(all_rec) > 0:
        real_rec = np.concatenate(all_rec)
        real_unc = np.concatenate(all_unc)
        real_labels = np.concatenate(all_labels)
        print(f"[✅ 成功加载 {dataset_name} 真实数据] 总样本数: {len(real_rec)}")
        return real_rec, real_unc, real_labels, True

    # 容错：占位数据预览
    print(f"[⚠️ 提示] 未找到 {dataset_name} 的完整真实结果，使用占位数据预览。")
    normal_rec = np.random.normal(loc=0.2, scale=0.1, size=50000)
    normal_unc = np.random.normal(loc=0.1, scale=0.05, size=50000)
    anomaly_rec = np.random.normal(loc=0.8, scale=0.2, size=3000)
    anomaly_unc = np.random.normal(loc=0.7, scale=0.2, size=3000)

    normal_rec, normal_unc = normal_rec[normal_rec > 0], normal_unc[normal_unc > 0]
    min_size = min(len(normal_rec), len(normal_unc))
    normal_rec, normal_unc = normal_rec[:min_size], normal_unc[:min_size]

    rec = np.concatenate([normal_rec, anomaly_rec])
    unc = np.concatenate([normal_unc, anomaly_unc])
    labels = np.concatenate([np.zeros(len(normal_rec)), np.ones(len(anomaly_rec))])

    return rec, unc, labels, False


# ==========================================
#  主绘图函数
# ==========================================
def plot_multi_dataset_joint_filtering():
    datasets = ['MSL', 'SMAP', 'SMD']

    # 🌟 修改点 2：将画布调整为竖向 (3行1列)，并设置合适的尺寸
    fig, axes = plt.subplots(3, 1, figsize=(12, 22))
    global_handles, global_labels = [], []

    for i, ds in enumerate(datasets):
        ax = axes[i]

        # 加载整个数据集
        rec_raw, unc_raw, labels, _ = load_full_dataset_data(ds)

        # 1. 使用鲁棒归一化 (基于整个数据集的分布)
        rec = robust_norm(rec_raw, p_max=99.0)
        unc = robust_norm(unc_raw, p_max=99.0)

        # 2. 增大不确定性的权重并计算联合得分
        lambda_weight = 1.5
        scores = rec + lambda_weight * unc

        # 3. 计算全局 IQR 阈值
        q1, q3 = np.percentile(scores, 25), np.percentile(scores, 75)
        iqr_thresh = q3 + 1.5 * (q3 - q1)

        # ==========================================================
        # 🌟 散点防爆卡顿保护机制
        # ==========================================================
        max_normal = 5000
        max_anomaly = 1000

        normal_idx = np.where(labels == 0)[0]
        anomaly_idx = np.where(labels == 1)[0]

        np.random.seed(42)
        if len(normal_idx) > max_normal:
            normal_idx = np.random.choice(normal_idx, max_normal, replace=False)
        if len(anomaly_idx) > max_anomaly:
            anomaly_idx = np.random.choice(anomaly_idx, max_anomaly, replace=False)

        # --- 绘制 Normal Samples ---
        h1 = ax.scatter(rec[normal_idx], unc[normal_idx],
                        c='#1f77b4', alpha=0.3, s=25, edgecolors='none',  # 点略微放大适配22号字
                        label='Normal samples')

        # --- 绘制 Anomaly Samples ---
        h2 = ax.scatter(rec[anomaly_idx], unc[anomaly_idx],
                        c='#d62728', marker='x', alpha=0.9, s=55, linewidth=2, zorder=5, # 点略微放大适配22号字
                        label='Anomaly samples')

        # ==========================================================
        # 🌟 阈值线绘制区
        # ==========================================================
        # --- 严格按照模型的 IQR 逻辑计算单一重构误差的截断线 ---
        q1_rec, q3_rec = np.percentile(rec, 25), np.percentile(rec, 75)
        only_rec_thresh = q3_rec + 1.5 * (q3_rec - q1_rec)
        only_rec_thresh = min(only_rec_thresh, 1.0)  # 防越界

        h4 = ax.axvline(x=only_rec_thresh, color='green', linestyle=':', linewidth=2.5, zorder=10,
                        label='Reconstruction error threshold')

        # --- 绘制联合阈值边界 (zorder=11) ---
        x_line = np.linspace(0, iqr_thresh, 100)
        y_line = (iqr_thresh - x_line) / lambda_weight
        h3, = ax.plot(x_line, y_line, color='black', linestyle='--', linewidth=2.5, zorder=11,
                      label='Joint threshold')

        # --- 完善图表元素 ---
        # 🌟 增大 title 的 pad 距离，防止与图例或者上一张图重叠
        ax.set_title(f'({chr(97 + i)}) {ds} Dataset', fontproperties=arial_font_title, pad=25)
        ax.set_xlabel('Reconstruction Error Perspective', fontproperties=arial_font)

        # 🌟 修改点 3：竖排模式下，每个子图都需要保留 Y 轴标签
        ax.set_ylabel('Uncertainty Estimation Perspective', fontproperties=arial_font)

        # 限制显示范围
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)

        # 抓取第一张图的所有图例元素
        if i == 0:
            global_handles, global_labels = ax.get_legend_handles_labels()

    # ---------------------------------------------------------
    #  🌟 绘制全局水平大图例 (Global Flattened Legend)
    # ---------------------------------------------------------
    by_label = dict(zip(global_labels, global_handles))

    # 🌟 修改点 4：
    # 1. 因为画布变窄，4个图例项排一行可能放不下，改为 ncol=2（分两行显示）
    # 2. bbox_to_anchor 下调至 1.02，让它与图像稍微靠近一点点
    fig.legend(by_label.values(), by_label.keys(), prop=arial_font_legend,
               loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, framealpha=0.9, edgecolor='black')

    # 🌟 修改点 5：调整布局，增大上下间距 (hspace=0.6)
    plt.tight_layout()
    plt.subplots_adjust(top=0.91, hspace=0.6)

    plt.savefig('Figure_Joint_Filtering_Vertical_V22.png', dpi=600, bbox_inches='tight')

    print("\n=====================================================")
    print("✅ 联合过滤特征图（竖排版+22号字体+大间距）已生成：Figure_Joint_Filtering_Vertical_V22.png")
    print("=====================================================")


if __name__ == '__main__':
    plot_multi_dataset_joint_filtering()