import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, precision_recall_curve, auc
from scipy.stats import wasserstein_distance  # 🌟 引入 W-Distance
import warnings

from exp.exp_anomaly_detection import Exp_Anomaly_Detection

warnings.filterwarnings('ignore')

# ==========================================
#  1. 基础配置与画图参数
# ==========================================
plt.rcdefaults()
plt.rcParams['font.sans-serif'] = ['Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
arial_font = fm.FontProperties(family='sans-serif', size=13)
arial_font_title = fm.FontProperties(family='sans-serif', size=15, weight='bold')

ROOT_PATH = "../数据集"
DATASET = "SMAP"  # 使用反差最大的数据集
SRC_ID = "A-7"
TRG_ID = "P-4"

SOURCE_ONLY_MODEL_PATH = f"./configs/{DATASET}/checkpoints_{SRC_ID}_to_{SRC_ID}/checkpoint.pth"
RADA_MODEL_PATH = f"./configs/{DATASET}/checkpoints_{SRC_ID}_to_{TRG_ID}/checkpoint.pth"


# ==========================================
#  2. 统计学距离计算函数 (Cohen's d)
# ==========================================
def compute_cohens_d(x, y):
    """计算效应量 (Cohen's d)，衡量两组一维得分的均值差异程度"""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    dof = nx + ny - 2
    # 汇合标准差 (Pooled Standard Deviation)
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    return np.abs(np.mean(x) - np.mean(y)) / (pooled_std + 1e-8)


# ==========================================
#  3. 特征提取器 (同时提取特征和重构得分)
# ==========================================
def extract_features(model, dataloader, device, max_samples=1500):
    model.eval()
    all_feats, all_scores, all_labels = [], [], []
    hook_outputs = []

    def hook(m, i, o):
        hook_outputs.append(o.detach().cpu().numpy())

    backbone = getattr(model, 'backbone_type', 'TCN')
    hook_layer = model.encoder_trans if backbone == 'Transformer' else model.encoder_cnn
    handle = hook_layer.register_forward_hook(hook)

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_size = batch_x.shape[0]
            batch_x = batch_x.float().to(device)
            hook_outputs.clear()

            try:
                score, _, _, _ = model.infer(batch_x, None, None, None)
            except ValueError:
                score, _ = model.infer(batch_x, None, None, None)

            score = score.detach().cpu().numpy().reshape(batch_size, -1).mean(axis=1)
            window_label = batch_y.numpy().reshape(batch_size, -1).max(axis=1)

            feat = hook_outputs[-1]
            feat = feat.reshape(batch_size, -1)  # Flatten 提纯

            all_feats.append(feat)
            all_scores.append(score)
            all_labels.append(window_label)

            if sum(len(f) for f in all_feats) >= max_samples:
                break

    handle.remove()
    return np.concatenate(all_feats)[:max_samples], np.concatenate(all_scores)[:max_samples], np.concatenate(
        all_labels)[:max_samples]


# ==========================================
#  4. 主画图流程
# ==========================================
class DummyArgs:
    def __init__(self):
        self.mode = 'test'
        self.configs_path = './configs/'
        self.save_path = './test_results/'
        self.root_path = ROOT_PATH
        self.data = DATASET
        self.data_origin = 'DADA'
        self.id_src = SRC_ID
        self.id_trg = TRG_ID
        self.iqr_mult = 1.0
        self.use_gpu = True
        self.gpu = 0
        self.gpu_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_multi_gpu = False
        self.devices = '0'
        self.method = 'spot'
        self.t = [0.1]
        self.metrics = ['best_f1']


def plot_tsne_metrics():
    args = DummyArgs()
    exp = Exp_Anomaly_Detection(args, id=0)
    model = exp.model
    device = exp.device

    print("📦 正在加载数据...")
    src_set, src_loader = exp._get_data(flag='train', entity_id=SRC_ID)
    tgt_train_set, tgt_train_loader = exp._get_data(flag='train', entity_id=TRG_ID)
    tgt_test_set, tgt_test_loader = exp._get_data(flag='test', entity_id=TRG_ID)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    model_paths = [SOURCE_ONLY_MODEL_PATH, RADA_MODEL_PATH]

    # 用于全局锁定坐标轴
    global_x_min, global_x_max = float('inf'), float('-inf')
    global_y_min, global_y_max = float('inf'), float('-inf')

    for idx, path in enumerate(model_paths):
        if not os.path.exists(path):
            print(f"⚠️ 找不到模型 {path}！")
            continue

        print(f"\n=============================================")
        print(f"🧠 分析模型: {'(a) Source Only' if idx == 0 else '(b) RADA'}")
        print(f"=============================================")
        model.load_state_dict(torch.load(path, map_location=device), strict=False)

        src_feats, src_scores, _ = extract_features(model, src_loader, device, max_samples=3000)
        tgt_train_feats, tgt_train_scores, _ = extract_features(model, tgt_train_loader, device, max_samples=4000)

        q1, q3 = np.percentile(tgt_train_scores, [25, 75])
        threshold = q3 + args.iqr_mult * (q3 - q1)

        mask_retained = tgt_train_scores <= threshold
        mask_filtered = tgt_train_scores > threshold

        tgt_retained_feats = tgt_train_feats[mask_retained]
        tgt_filtered_feats = tgt_train_feats[mask_filtered]

        tgt_test_feats, tgt_test_scores, tgt_test_labels = extract_features(model, tgt_test_loader, device,
                                                                            max_samples=8000)

        # 提取目标域测试集中的真实正常点和异常点
        true_normal_mask = tgt_test_labels == 0
        true_anom_mask = tgt_test_labels == 1

        true_anomaly_feats = tgt_test_feats[true_anom_mask]
        if len(true_anomaly_feats) > 600:
            true_anomaly_feats = true_anomaly_feats[np.random.choice(len(true_anomaly_feats), 200, replace=False)]

        all_feats = np.vstack([src_feats, tgt_retained_feats, tgt_filtered_feats, true_anomaly_feats])
        n_samples = all_feats.shape[0]
        pca_dim = min(50, n_samples)

        pca = PCA(n_components=pca_dim, random_state=42)
        all_feats_pca = pca.fit_transform(all_feats)

        idx_src = len(src_feats)
        idx_tgt_ret = idx_src + len(tgt_retained_feats)
        idx_tgt_fil = idx_tgt_ret + len(tgt_filtered_feats)

        feats_src_pca = all_feats_pca[:idx_src]
        feats_tgt_ret_pca = all_feats_pca[idx_src:idx_tgt_ret]
        feats_tgt_anom_pca = all_feats_pca[idx_tgt_fil:]

        # --------------------------------------------------
        # 🌟 核心指标计算区 (4大分离度神级指标)
        # --------------------------------------------------
        w_dist = 0.0
        c_d = 0.0
        anom_sil = 0.0
        latent_aupr = 0.0

        # 指标用全量数据，单独提取（不参与 t-SNE）
        tgt_test_feats_full, tgt_test_scores_full, tgt_test_labels_full = extract_features(
            model, tgt_test_loader, device, max_samples=999999)

        norm_scores = tgt_test_scores_full[tgt_test_labels_full == 0]
        anom_scores = tgt_test_scores_full[tgt_test_labels_full == 1]

        if len(norm_scores) > 0 and len(anom_scores) > 0:
            # 1. Wasserstein Distance (基于重构得分)
            w_dist = wasserstein_distance(norm_scores, anom_scores)

            # 2. Cohen's d 效应量 (基于重构得分)
            c_d = compute_cohens_d(norm_scores, anom_scores)

        if len(feats_tgt_ret_pca) > 0 and len(feats_tgt_anom_pca) > 0:
            # 3. Anomaly Silhouette (基于 Latent Space)
            anom_labels = np.concatenate([np.zeros(len(feats_tgt_ret_pca)), np.ones(len(feats_tgt_anom_pca))])
            anom_sil = silhouette_score(np.vstack([feats_tgt_ret_pca, feats_tgt_anom_pca]), anom_labels)

            # 4. Latent Space AUPR (依靠到源域中心的几何距离)
            src_center = np.mean(feats_src_pca, axis=0)
            dist_normal = np.linalg.norm(feats_tgt_ret_pca - src_center, axis=1)
            dist_anom = np.linalg.norm(feats_tgt_anom_pca - src_center, axis=1)

            latent_scores = np.concatenate([dist_normal, dist_anom])
            precision, recall, _ = precision_recall_curve(anom_labels, latent_scores)
            latent_aupr = auc(recall, precision)

        print(f"✅ Wasserstein Dist: {w_dist:.4f} | Cohen's d: {c_d:.4f}")
        print(f"✅ Anomaly Silhouette: {anom_sil:.4f} | Latent AUPR: {latent_aupr:.4f}")

        # 将 4 个指标通过换行符排版在标题上
        title = (f"(a) Source Only\n" if idx == 0 else f"(b) RADA\n")
        title += f"W-Distance: {w_dist:.2f}  |  Cohen's d: {c_d:.2f}\n"
        title += f"Anomaly Silhouette: {anom_sil:.4f}  |  Latent AUPR: {latent_aupr:.4f}"

        print("🌌 正在进行 t-SNE 聚类...")
        dynamic_perplexity = min(40, max(5, n_samples // 4))
        tsne = TSNE(n_components=2, perplexity=dynamic_perplexity, max_iter=1000, init='pca', learning_rate='auto',
                    random_state=42)
        all_tsne = tsne.fit_transform(all_feats_pca)

        tsne_src = all_tsne[:idx_src]
        tsne_tgt_ret = all_tsne[idx_src:idx_tgt_ret]
        tsne_tgt_fil = all_tsne[idx_tgt_ret:idx_tgt_fil]
        tsne_anom = all_tsne[idx_tgt_fil:]

        global_x_min, global_x_max = min(global_x_min, all_tsne[:, 0].min()), max(global_x_max, all_tsne[:, 0].max())
        global_y_min, global_y_max = min(global_y_min, all_tsne[:, 1].min()), max(global_y_max, all_tsne[:, 1].max())

        ax = axes[idx]
        ax.scatter(tsne_src[:, 0], tsne_src[:, 1], c='#1f77b4', label='Source Domain (Normal)', alpha=0.3, s=20,
                   zorder=2)
        ax.scatter(tsne_tgt_ret[:, 0], tsne_tgt_ret[:, 1], c='#2ca02c', label='Target Domain (Pseudo-Normal)',
                   alpha=0.3, s=20, zorder=3)
        ax.scatter(tsne_tgt_fil[:, 0], tsne_tgt_fil[:, 1], c='#ff7f0e', label='Target Domain (Filtered Outliers)',
                   alpha=0.9, s=40, edgecolors='white', zorder=4)
        if len(tsne_anom) > 0:
            ax.scatter(tsne_anom[:, 0], tsne_anom[:, 1], c='#d62728', label='Target Domain (True Anomalies)',
                       marker='X', s=80, edgecolors='black', linewidth=0.8, zorder=5)

        # 增加 pad 以免三行标题和图表重叠
        ax.set_title(title, fontproperties=arial_font_title, pad=20, linespacing=1.5)
        ax.set_xticks([])
        ax.set_yticks([])

        if idx == 1:
            ax.legend(prop=arial_font, loc='upper left', bbox_to_anchor=(1.05, 1), framealpha=0.9, edgecolor='black')

    # 强锁取景框：两张图的 X 轴和 Y 轴尺度绝对一致！
    pad_x = (global_x_max - global_x_min) * 0.05
    pad_y = (global_y_max - global_y_min) * 0.05
    for ax in axes:
        ax.set_xlim(global_x_min - pad_x, global_x_max + pad_x)
        ax.set_ylim(global_y_min - pad_y, global_y_max + pad_y)

    plt.tight_layout()
    print(f"src样本数: {len(src_set)}")
    print(f"tgt_train样本数: {len(tgt_train_set)}")
    print(f"tgt_test样本数: {len(tgt_test_set)}")
    plt.savefig('Figure_tSNE_Separation_Metrics.png', dpi=600, bbox_inches='tight')
    # plt.savefig('Figure_tSNE_Separation_Metrics.pdf', format='pdf', bbox_inches='tight')
    print("\n✅ 终极分离度可视化图表已生成：Figure_tSNE_Separation_Metrics.png")



if __name__ == "__main__":
    plot_tsne_metrics()