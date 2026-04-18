import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 导入你的模型和真实的 NeurIPS-TS 数据生成器
from models.AMCAD.Basic_AMCAD import Basic_AMCAD
from data_provider.Generator.multivariate_generator import MultivariateDataGenerator, sine, cosine


class MockConfigs:
    """模拟 args/configs，提供模型初始化所需的超参数"""

    def __init__(self, seq_len=100, d_model=64, backbone_type='TCN'):
        self.seq_len = seq_len
        self.patch_len = 10
        self.d_model = d_model
        self.backbone_type = backbone_type
        self.ms_kernels = [3, 5, 7]
        self.ms_method = "interval_sampling"
        self.e_layers = 3
        self.d_layers = 3


def get_neurips_synthetic_data(length, dim, save_path):
    """使用 NeurIPS-TS 生成真实合成数据，并保存到本地硬盘"""
    # 交替使用 sine 和 cosine 构造不同维度的基信号
    behavior = [sine if i % 2 == 0 else cosine for i in range(dim)]
    behavior_config = [{'freq': 0.04, 'coef': 1.5, "offset": 0.0, 'noise_amp': 0.05} for _ in range(dim)]

    # 1. 实例化数据生成器并生成数据
    generator = MultivariateDataGenerator(dim=dim, stream_length=length,
                                          behavior=behavior, behavior_config=behavior_config)

    # 2. 注入异常 (复现真实场景)
    generator.point_global_outliers(dim_no=0, ratio=0.01, factor=3.5, radius=5)
    if dim > 1:
        generator.collective_trend_outliers(dim_no=1, ratio=0.01, factor=0.5, radius=5)

    # 3. 获取数据并保存
    # generator.data 形状是 [dim, length]，我们需要转置为 [length, dim]
    ts_data = generator.data.T
    labels = generator.label

    # 将数据和标签拼接后保存到本地文件夹
    save_array = np.concatenate([ts_data, labels.reshape(-1, 1)], axis=1)
    np.save(save_path, save_array)

    return torch.tensor(ts_data, dtype=torch.float32)


class RealSyntheticDataLoader:
    """滑动窗口数据加载器"""

    def __init__(self, full_tensor, seq_len, batch_size):
        self.full_tensor = full_tensor
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.total_length = full_tensor.shape[0]

        num_samples = self.total_length - seq_len + 1
        self.num_batches = num_samples // batch_size
        self.valid_samples = self.num_batches * batch_size

        indices = torch.arange(self.valid_samples).unsqueeze(1) + torch.arange(seq_len)
        self.dataset = self.full_tensor[indices]  # [valid_samples, seq_len, dim]

    def __iter__(self):
        for i in range(self.num_batches):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size
            yield self.dataset[start_idx:end_idx]


def measure_efficiency(model, data_loader, device):
    """测量总执行时间和峰值显存"""
    model.eval()

    # 预热
    dummy_x = next(iter(data_loader)).to(device)
    with torch.no_grad():
        for _ in range(3):
            model.infer(dummy_x)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        for batch_x in data_loader:
            batch_x = batch_x.to(device)
            _ = model.infer(batch_x)

    end_event.record()
    torch.cuda.synchronize()

    exec_time = start_event.elapsed_time(end_event) / 1000.0
    max_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return exec_time, max_memory_mb


def plot_efficiency_results(df_dim, df_len):
    """严格按照论文标准绘制带有对数刻度的效率图"""
    print("正在生成效率曲线图...")
    df_dim_valid = df_dim[df_dim['Time(s)'] != 'OOM'].copy()
    df_dim_valid['Time(s)'] = df_dim_valid['Time(s)'].astype(float)

    df_len_valid = df_len[df_len['Time(s)'] != 'OOM'].copy()
    df_len_valid['Time(s)'] = df_len_valid['Time(s)'].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ================= 图 1: Time vs Dimensionality =================
    ax1 = axes[0]
    ax1.plot(df_dim_valid['Dimension'], df_dim_valid['Time(s)'], marker='o', linestyle='-', color='b')

    ax1.set_xscale('log', base=2)
    dim_ticks = [8, 16, 32, 64, 128, 256, 512, 1024]
    ax1.set_xticks(dim_ticks)
    ax1.set_xticklabels(dim_ticks)

    ax1.set_yscale('log', base=10)
    y_dim_ticks = [1, 10, 100]
    ax1.set_yticks(y_dim_ticks)
    ax1.set_yticklabels(['10^0', '10^1', '10^2'])

    ax1.set_xlabel('Dimensionality', fontsize=12)
    ax1.set_ylabel('Execution time (second)', fontsize=12)
    ax1.set_title('Time vs Dimensionality', fontsize=14)
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # ================= 图 2: Time vs Length =================
    ax2 = axes[1]
    ax2.plot(df_len_valid['Length'], df_len_valid['Time(s)'], marker='s', linestyle='-', color='r')

    ax2.set_xscale('log', base=2)
    len_ticks = [2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]
    ax2.set_xticks(len_ticks)
    ax2.set_xticklabels(len_ticks, rotation=45)

    ax2.set_yscale('log', base=10)
    y_len_ticks = [1, 10, 100, 1000, 10000]
    ax2.set_yticks(y_len_ticks)
    ax2.set_yticklabels(['10^0', '10^1', '10^2', '10^3', '10^4'])

    ax2.set_xlabel('Length', fontsize=12)
    ax2.set_ylabel('Execution time (second)', fontsize=12)
    ax2.set_title('Time vs Length', fontsize=14)
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    save_path = 'efficiency_plots.png'
    plt.savefig(save_path, dpi=300)
    print(f"图表已保存至: {os.path.abspath(save_path)}")


def run_experiments():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # 创建保存合成数据的文件夹
    data_save_dir = "synthetic_efficiency_data"
    os.makedirs(data_save_dir, exist_ok=True)
    print(f"所有生成的数据集将被保存在: ./{data_save_dir}/ 文件夹下\n")

    seq_len = 100
    batch_size = 128

    configs = MockConfigs(seq_len=seq_len, d_model=64, backbone_type='TCN')
    model = Basic_AMCAD(configs).to(device)

    results_dim = []
    results_len = []

    # ================= 实验 1: 维度扩展测试 =================
    print("-" * 50)
    print("实验 1: 维度扩展测试 (固定长度 L=2000)")
    print("-" * 50)
    fixed_length = 2000
    dims = [8, 16, 32, 64, 128, 256, 512, 1024]

    for dim in dims:
        torch.cuda.empty_cache()
        save_file = os.path.join(data_save_dir, f"dim_test_L{fixed_length}_D{dim}.npy")

        full_tensor = get_neurips_synthetic_data(length=fixed_length, dim=dim, save_path=save_file)
        loader = RealSyntheticDataLoader(full_tensor, seq_len=configs.seq_len, batch_size=batch_size)

        try:
            exec_time, peak_mem = measure_efficiency(model, loader, device)
            results_dim.append({'Dimension': dim, 'Length': fixed_length, 'Time(s)': exec_time, 'Memory(MB)': peak_mem})
            print(f"Dim: {dim:<5} | Time: {exec_time:.4f} s | Memory: {peak_mem:.2f} MB")
        except RuntimeError as e:
            if "Out of memory" in str(e):
                print(f"Dim: {dim:<5} | Time: OOM | Memory: OOM")
                results_dim.append({'Dimension': dim, 'Length': fixed_length, 'Time(s)': 'OOM', 'Memory(MB)': 'OOM'})
            else:
                raise e

    # ================= 实验 2: 序列长度扩展测试 =================
    print("\n" + "-" * 50)
    print("实验 2: 序列长度扩展测试 (固定维度 D=8)")
    print("-" * 50)
    fixed_dim = 8
    lengths = [2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]

    for length in lengths:
        torch.cuda.empty_cache()
        save_file = os.path.join(data_save_dir, f"len_test_D{fixed_dim}_L{length}.npy")

        full_tensor = get_neurips_synthetic_data(length=length, dim=fixed_dim, save_path=save_file)
        loader = RealSyntheticDataLoader(full_tensor, seq_len=configs.seq_len, batch_size=batch_size)

        try:
            exec_time, peak_mem = measure_efficiency(model, loader, device)
            results_len.append({'Dimension': fixed_dim, 'Length': length, 'Time(s)': exec_time, 'Memory(MB)': peak_mem})
            print(f"Length: {length:<7} | Time: {exec_time:.4f} s | Memory: {peak_mem:.2f} MB")
        except RuntimeError as e:
            if "Out of memory" in str(e):
                print(f"Length: {length:<7} | Time: OOM | Memory: OOM")
                results_len.append({'Dimension': fixed_dim, 'Length': length, 'Time(s)': 'OOM', 'Memory(MB)': 'OOM'})
            else:
                raise e

    # 保存日志并绘图
    pd.DataFrame(results_dim).to_csv('efficiency_dimension_test.csv', index=False)
    pd.DataFrame(results_len).to_csv('efficiency_length_test.csv', index=False)
    plot_efficiency_results(pd.DataFrame(results_dim), pd.DataFrame(results_len))
    print("\n所有实验及绘图已完成！")


if __name__ == "__main__":
    run_experiments()