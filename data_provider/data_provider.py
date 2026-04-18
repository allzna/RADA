import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
import ast

warnings.filterwarnings('ignore')


def data_provider(root_path, datasets, batch_size, win_size=100, step=100, flag="train", percentage=1, entity_id=None,
                  do_normalization=True, pre_loaded_data=None, **kwargs):

    if flag == "train":
        shuffle = True
    else:
        shuffle = False

    train_stats = kwargs.get('train_stats', None)
    use_test_data = kwargs.get('use_test_data', False)

    loader_info = f"loading {datasets}"
    if entity_id:
        loader_info += f" [Entity: {entity_id}]"

    if pre_loaded_data is not None:
        print(f"{loader_info}({flag}) [Using Pre-loaded Data] percentage: {percentage * 100}% ...", end="")
        file_paths = None  # 不需要路径
    else:
        print(f"{loader_info}({flag}) percentage: {percentage * 100}% ...", end="")

        file_paths = None
        if datasets == 'HHAR_P':
            prefix = 'train' if flag == 'train' else 'test'
            if use_test_data: prefix = 'test'

            possible_paths = [
                os.path.join(root_path, datasets, f"{prefix}_{entity_id}.pt"),
                os.path.join(root_path, "data", datasets, f"{prefix}_{entity_id}.pt"),
                os.path.join(root_path, f"{prefix}_{entity_id}.pt")
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    file_paths = p
                    break
            if file_paths is None:
                raise FileNotFoundError(f"Could not find HHAR_P data for entity {entity_id} (flag={flag})")

        elif entity_id:
            search_type = 'train'
            if flag == 'test' or use_test_data:
                search_type = 'test'

            possible_paths = []

            if search_type == 'train':
                possible_paths = [
                    os.path.join(root_path, datasets, entity_id, "train.csv"),
                    os.path.join(root_path, datasets, entity_id, "test.csv"),
                    os.path.join(root_path, datasets, "train", f"machine-{entity_id}.txt"),
                    os.path.join(root_path, datasets, "train", f"{entity_id}.txt"),
                    os.path.join(root_path, "MSL_SMAP", "train", f"{entity_id}.npy"),
                    os.path.join(root_path, "MSL_SMAP", f"{entity_id}_train.npy"),
                    os.path.join(root_path, "data", datasets, "train", f"{entity_id}.npy"),
                    os.path.join(root_path, datasets, f"machine-{entity_id}.txt"),
                    os.path.join(root_path, datasets, f"{entity_id}.txt"),
                ]
            else:
                possible_paths = [
                    os.path.join(root_path, datasets, entity_id, "test.csv"),
                    os.path.join(root_path, datasets, entity_id, "train.csv"),
                    os.path.join(root_path, datasets, "test", f"machine-{entity_id}.txt"),
                    os.path.join(root_path, datasets, "test", f"{entity_id}.txt"),
                    os.path.join(root_path, "MSL_SMAP", "test", f"{entity_id}.npy"),
                    os.path.join(root_path, "MSL_SMAP", f"{entity_id}.npy"),
                    os.path.join(root_path, "data", datasets, "test", f"{entity_id}.npy"),
                    os.path.join(root_path, datasets, "test", f"{entity_id}.npy"),
                    os.path.join(root_path, datasets, f"{entity_id}.npy"),
                ]

            for p in possible_paths:
                if os.path.exists(p):
                    file_paths = p
                    break

            if file_paths is None and datasets != 'HHAR_P':
                try:
                    base_path, _ = read_meta(root_path=root_path, dataset=datasets)
                    if base_path and os.path.isdir(base_path):
                        file_paths = os.path.join(base_path, f"{entity_id}.csv")
                except:
                    pass

            if file_paths is None:
                raise FileNotFoundError(
                    f"Could not find data file for entity {entity_id} in {datasets} (search_type={search_type})")

        else:
            file_paths, _ = read_meta(root_path=root_path, dataset=datasets)

    train_lens = None
    discrete_channels = None
    if datasets == "MSL": discrete_channels = range(1, 55)
    if datasets == "SMAP": discrete_channels = range(1, 25)
    if datasets == "SWAT": discrete_channels = [2, 4, 9, 10, 11, 13, 15, 19, 20, 21, 22, 29, 30, 31, 32, 33, 42, 43, 48,
                                                50]

    # 将 parameter 传递给 Dataset
    data_set = TrainSegLoader(file_paths, train_lens, win_size, step, flag, percentage, discrete_channels,
                              entity_id=entity_id, dataset_name=datasets, train_stats=train_stats,
                              do_normalization=do_normalization, pre_loaded_data=pre_loaded_data)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    print("done!")
    return data_set, data_loader


class TrainSegLoader(Dataset):
    def __init__(self, data_path, train_length, win_size, step, flag="train", percentage=1, discrete_channels=None,
                 entity_id=None, dataset_name=None, train_stats=None, do_normalization=True, pre_loaded_data=None):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.entity_id = entity_id

        # 读取数据
        if pre_loaded_data is not None:
            data = pre_loaded_data
        else:
            data = read_data(data_path, entity_id, dataset_name=dataset_name)

        # 预处理
        raw_data = data.iloc[:, :]
        features, labels = (
            raw_data.loc[:, raw_data.columns != "label"].to_numpy(),
            raw_data.loc[:, ["label"]].to_numpy(),
        )

        if discrete_channels is not None:
            if features.shape[1] > max(discrete_channels):
                features = np.delete(features, discrete_channels, axis=-1)

        # 归一化
        train_end = int(len(features) * 0.9)
        self.scaler = StandardScaler()

        if do_normalization:
            if flag == 'train' or train_stats is None:
                # 仅使用训练集部分 fit
                train_feat_for_fit = features[:train_end]
                train_label_for_fit = labels[:train_end]

                # 仅在正常数据上计算统计量
                normal_mask = (train_label_for_fit.flatten() == 0)
                if sum(normal_mask) > 0:
                    self.scaler.fit(train_feat_for_fit[normal_mask])
                else:
                    self.scaler.fit(train_feat_for_fit)
            else:
                # 使用传入的统计量
                self.scaler.mean_ = train_stats['mean']
                self.scaler.scale_ = train_stats['std']
                self.scaler.var_ = train_stats['std'] ** 2

            norm_features = self.scaler.transform(features)
        else:
            # 不归一化时直接使用原始特征
            norm_features = features

        # 划分数据集 & 恢复属性名

        if flag == "train":
            self.train = norm_features[:train_end]
            self.train_label = labels[:train_end]
            self.data = self.train
            self.label = self.train_label

            # Boiler 专用清洗逻辑
            self.valid_indices = []
            possible_starts = range(0, len(self.data) - self.win_size + 1, self.step)

            if dataset_name == 'Boiler':
                for start in possible_starts:
                    end = start + self.win_size
                    window_labels = self.label[start:end]
                    # 只保留纯净窗口
                    if np.sum(window_labels) == 0:
                        self.valid_indices.append(start)
                print(f"[Info] Boiler Cleaning: Generated {len(self.valid_indices)} pure normal windows.")
                if len(self.valid_indices) == 0:
                    # 如果清洗太干净导致没数据了，回退到全部数据（防止报错）
                    print("[Warning] No pure windows found! Fallback to all windows.")
                    self.valid_indices = list(possible_starts)
            else:
                self.valid_indices = list(possible_starts)

        elif flag == "val":
            self.val = norm_features[train_end:]
            self.val_label = labels[train_end:]
            self.data = self.val
            self.label = self.val_label
            self.valid_indices = list(range(0, len(self.data) - self.win_size + 1, self.step))

        elif flag == "test":
            self.test = norm_features
            self.test_label = labels
            self.data = self.test
            self.label = self.test_label
            self.valid_indices = list(range(0, len(self.data) - self.win_size + 1, self.step))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        # 使用清洗后的索引列表
        start_idx = self.valid_indices[index]
        end_idx = start_idx + self.win_size

        seq = np.float32(self.data[start_idx: end_idx])
        lbl = np.float32(self.label[start_idx: end_idx])

        return seq, lbl


def read_data(path: str, entity_id=None, nrows=None, dataset_name=None) -> pd.DataFrame:
    df = pd.DataFrame()

    if path.endswith('.pt'):
        try:
            dataset = torch.load(path)
            if isinstance(dataset, dict):
                X = dataset.get('samples')
                y = dataset.get('labels')
            else:
                try:
                    X, y = dataset
                except:
                    X = dataset;
                    y = torch.zeros(X.shape[0])
            if isinstance(X, np.ndarray): X = torch.from_numpy(X)
            if isinstance(y, np.ndarray): y = torch.from_numpy(y)
            if len(X.shape) == 3 and X.shape[1] < X.shape[2]: X = X.permute(0, 2, 1)
            num_channels = X.shape[2];
            seq_len = X.shape[1]
            X_flat = X.reshape(-1, num_channels).numpy()
            if len(y.shape) == 1:
                y_flat = y.unsqueeze(1).repeat(1, seq_len).reshape(-1).numpy()
            else:
                y_flat = y.reshape(-1).numpy()
            df = pd.DataFrame(X_flat);
            df['label'] = y_flat
        except Exception as e:
            print(f"[Error] Failed reading pt file: {e}")
            return pd.DataFrame()
        return df

    elif path.endswith('.npy'):
        try:
            data = np.load(path)
            if np.any(np.isnan(data)): data = np.nan_to_num(data)
            df = pd.DataFrame(data)
            if dataset_name in ['MSL', 'SMAP'] and entity_id is not None:
                df["label"] = 0
                parent_dir = os.path.dirname(path)
                grandparent_dir = os.path.dirname(parent_dir)
                csv_path = None
                if os.path.exists(os.path.join(parent_dir, 'labeled_anomalies.csv')):
                    csv_path = os.path.join(parent_dir, 'labeled_anomalies.csv')
                elif os.path.exists(os.path.join(grandparent_dir, 'labeled_anomalies.csv')):
                    csv_path = os.path.join(grandparent_dir, 'labeled_anomalies.csv')
                if csv_path:
                    try:
                        label_df = pd.read_csv(csv_path)
                        row = label_df[label_df['chan_id'] == entity_id]
                        if not row.empty:
                            anomaly_str = row.iloc[0]['anomaly_sequences']
                            anomalies = ast.literal_eval(anomaly_str)
                            for start, end in anomalies:
                                start = max(0, int(start))
                                end = min(len(df) - 1, int(end))
                                df.loc[start:end, "label"] = 1
                    except Exception as e:
                        pass
            else:
                df["label"] = 0
            return df
        except Exception as e:
            print(f"[Error] Failed reading npy file: {e}")
            return pd.DataFrame()

    else:
        try:
            if dataset_name == 'SMD':
                data = pd.read_csv(path, header=None)
            else:
                data = pd.read_csv(path)

            df = data.copy()

            if dataset_name == 'Boiler':
                if 'Abnormal Blow Down' in df.columns:
                    df['label'] = df['Abnormal Blow Down']
                    # 增加要剔除的常数/状态列
                    # 除了 boiler_no, TIME, label 外，把那些常数列也加进去
                cols_to_drop = [
                    'boiler_no', 'TIME', 'Abnormal Blow Down',
                    'Operating Status', 'Operating Code', 'Input Status',
                    'Operating Time_Feed water', 'Operating Time_Chemical Injection',
                    'Combustion time', 'Number of Ignition'
                ]
                df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

            if dataset_name == 'SMD' and 'test' in path:
                try:
                    dir_name = os.path.dirname(path)
                    file_name = os.path.basename(path)
                    if '/test' in dir_name or '\\test' in dir_name:
                        label_dir = dir_name.replace('/test', '/test_label').replace('\\test', '\\test_label')
                        label_path = os.path.join(label_dir, file_name)
                        if os.path.exists(label_path):
                            label_df = pd.read_csv(label_path, header=None)
                            if len(label_df) == len(df):
                                df['label'] = label_df.iloc[:, 0].values
                            else:
                                print(f"[ERROR] 长度不匹配! 数据长度: {len(df)}, 标签长度: {len(label_df)}")
                except Exception as e:
                    print(f"[ERROR] 加载标签时发生异常: {e}")

            if "label" not in df.columns:
                if "Label" in df.columns:
                    df["label"] = df["Label"]
                elif "anomaly" in df.columns:
                    df["label"] = df["anomaly"]
                else:
                    df["label"] = 0

            return df

        except Exception as e:
            print(f"[Error] Reading CSV failed: {e}")
            return pd.DataFrame()


def read_meta(root_path, dataset):
    meta_path = root_path + "/DETECT_META.csv"
    if not os.path.exists(meta_path):
        return None, None

    meta = pd.read_csv(meta_path)
    meta = meta.query(f'file_name.str.contains("{dataset}")', engine="python")
    if meta.empty:
        return None, None

    file_paths = root_path + f"/data/{meta.file_name.values[0]}"
    train_lens = meta.train_lens.values[0]
    return file_paths, train_lens