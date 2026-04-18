import torch.multiprocessing

# torch.multiprocessing.set_sharing_strategy('file_system') # Windows下可能需要注释
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np
import math
import json
import pandas as pd
import logging
from data_provider.data_provider import data_provider

import sys
import random

try:
    from augmentations import Injector
except ImportError:

    pass

from models.AMCAD import Basic_AMCAD

warnings.filterwarnings('ignore')


# 辅助类定义 (移至顶部以避免 NameError)

class Configs:
    def __init__(self, json_path):
        with open(json_path) as f:
            configs = json.load(f)
            self.__dict__.update(configs)


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, logger=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.logger = logger

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                msg = f'EarlyStopping counter: {self.counter} out of {self.patience}'
                if self.logger:
                    self.logger.info(msg)
                else:
                    print(msg)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            msg = f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss



#  实验主类


class Exp_Anomaly_Detection():
    def __init__(self, args, id=None):
        self.args = args
        self.device = self._acquire_device()

        # 路径与后缀设置
        self.use_uda = hasattr(self.args, 'id_src') and self.args.id_src is not None and \
                       hasattr(self.args, 'id_trg') and self.args.id_trg is not None

        if self.use_uda:
            exp_suffix = f"_{self.args.id_src}_to_{self.args.id_trg}"
            config_id = id
        else:
            exp_suffix = f"_{id}"
            config_id = id

        if self.args.data_origin == "UCR":
            self.id = id
            self.model_configs = Configs(
                os.path.join(self.args.configs_path, self.args.data_origin, f"model_configs_{config_id}.json"))
            self.train_configs = Configs(
                os.path.join(self.args.configs_path, self.args.data_origin, "train_configs.json"))

            base_dir = self.args.data_origin
            sub_dir = self.args.data
        else:
            self.model_configs = Configs(
                os.path.join(self.args.configs_path, self.args.data, f"model_configs_{config_id}.json"))
            self.train_configs = Configs(
                os.path.join(self.args.configs_path, self.args.data, "train_configs.json"))

            base_dir = self.args.data
            sub_dir = ""
            self.model_configs.iqr_mult = self.args.iqr_mult

        if self.args.data_origin == "UCR":
            self.model_save_path = os.path.join(self.args.configs_path, base_dir, sub_dir, f"checkpoints{exp_suffix}")
            self.rst_save_path = os.path.join(self.args.save_path, base_dir, sub_dir, f"results{exp_suffix}")
        else:
            self.model_save_path = os.path.join(self.args.configs_path, base_dir, f"checkpoints{exp_suffix}")
            self.rst_save_path = os.path.join(self.args.save_path, base_dir, f"results{exp_suffix}")

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        if not os.path.exists(self.rst_save_path):
            os.makedirs(self.rst_save_path)

        # 初始化 Logger
        self.logger = self._build_logger(self.model_save_path)
        self.logger.info("-" * 50)
        self.logger.info(f"Experiment started.")
        self.logger.info(f"Dataset: {self.args.data}, Origin: {self.args.data_origin}")
        self.logger.info(f"Mode: {'UDA (Adaptation + Anchored Memory)' if self.use_uda else 'Standard AD'}")
        if self.use_uda:
            self.logger.info(f"Source: {self.args.id_src} -> Target: {self.args.id_trg}")
        self.logger.info(f"Checkpoints Path: {self.model_save_path}")
        self.logger.info(f"Results Path: {self.rst_save_path}")

        # 保存配置
        self._save_configurations()

        self.model = self._build_model().to(self.device)

    def _build_logger(self, log_dir):
        logger = logging.getLogger(f"Exp_{self.args.data}_{time.time()}")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        log_file = os.path.join(log_dir, 'run.log')
        fh = logging.FileHandler(log_file, mode='a')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def _save_configurations(self):
        configs = {
            "args": vars(self.args),
            "model_configs": vars(self.model_configs),
            "train_configs": vars(self.train_configs)
        }
        if 'device' in configs['args']:
            configs['args']['device'] = str(configs['args']['device'])
        config_path = os.path.join(self.model_save_path, 'config_all.json')
        with open(config_path, 'w') as f:
            json.dump(configs, f, indent=4)
        self.logger.info(f"Configuration saved to {config_path}")

    def _build_model(self):
        model = Basic_AMCAD.Basic_AMCAD(self.model_configs)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _basic(self):
        from thop import profile
        from thop import clever_format
        dummy_input = torch.rand(1, self.model_configs.seq_len, 1).to(self.device)
        try:
            pass
        except Exception as e:
            self.logger.warning(f"Skipped FLOPs calculation: {e}")

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        return device

    def _get_data(self, flag, step=None, entity_id=None, use_test_data=False, train_stats=None, do_normalization=True,
                  pre_loaded_data=None):
        if self.args.data_origin == "DADA":
            from data_provider.data_provider import data_provider
        elif self.args.data_origin == "UCR":
            from data_provider.data_provider_UCR import data_provider
        else:
            from data_provider.data_provider import data_provider

        win_size = self.model_configs.seq_len
        if step is None:
            step = win_size
        batch_size = self.train_configs.batch_size

        data_set, data_loader = data_provider(
            root_path=self.args.root_path,
            datasets=self.args.data,
            batch_size=batch_size,
            win_size=win_size,
            step=step,
            flag=flag,
            entity_id=entity_id,
            use_test_data=use_test_data,
            device=self.device,
            train_stats=train_stats,
            do_normalization=do_normalization,
            pre_loaded_data=pre_loaded_data
        )

        return data_set, data_loader

    def _select_optimizer(self):
        if self.train_configs.optim == "adam":
            model_optim = optim.Adam(self.model.parameters(), lr=self.train_configs.learning_rate)
        elif self.train_configs.optim == "adamw":
            model_optim = optim.AdamW(self.model.parameters(), lr=self.train_configs.learning_rate)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.train_configs.learning_rate)
        return model_optim

    def _adjust_learning_rate(self, optimizer, epoch, train_configs, verbose=True, **other_args):
        if train_configs.lradj == 'type1':
            lr_adjust = {epoch: train_configs.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        elif train_configs.lradj == 'type2':
            lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
        elif train_configs.lradj == 'type3':
            lr_adjust = {epoch: train_configs.learning_rate if epoch < 3 else train_configs.learning_rate * (
                    0.9 ** ((epoch - 3) // 1))}
        elif train_configs.lradj == "cosine":
            lr_adjust = {
                epoch: train_configs.learning_rate / 2 * (1 + math.cos(epoch / train_configs.train_epochs * math.pi))}
        elif train_configs.lradj == '1cycle':
            scheduler = other_args['scheduler']
            lr_adjust = {epoch: scheduler.get_last_lr()[0]}

        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if verbose: self.logger.info('Updating learning rate to {}'.format(lr))

    def vali(self, vali_loader, use_uda=False):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                if use_uda:
                    # UDA 验证
                    recon_loss, latent_loss, _ = self.model(
                        batch_x, None, None, None,
                        domain='target', update_flag=False
                    )
                    loss = recon_loss + 0.1 * latent_loss
                else:
                    ms_loss, q_latent_distance, _ = self.model(
                        batch_x, None, None, None
                    )
                    loss = ms_loss

                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        self._basic()
        history = {'train_loss': [], 'vali_loss': []}

        if self.use_uda:
            id_trg = getattr(self.args, 'id_trg', None)
            step_size = 1 if self.args.data == 'SMD' else 1


            # 使用 Target Median/IQR 进行归一化


            # 加载源域
            src_set, src_loader = self._get_data(flag='train', step=step_size, entity_id=self.args.id_src)

            # 加载目标域全量数据
            self.logger.info(f"Calculating ROBUST statistics (Median/IQR) for Target Domain ({id_trg})...")

            temp_set, _ = self._get_data(flag='test', entity_id=id_trg, use_test_data=True, train_stats=None,
                                         do_normalization=False)

            raw_features = temp_set.test
            raw_labels = temp_set.test_label

            #计算鲁棒统计量
            target_robust_mean = np.median(raw_features, axis=0)
            q75, q25 = np.percentile(raw_features, [75, 25], axis=0)
            iqr = q75 - q25
            target_robust_std = iqr / 1.349

            zero_mask = target_robust_std < 1e-6
            if zero_mask.any():
                self.logger.warning(f"Found {zero_mask.sum()} constant features in Target. Fallback to Source Std.")
                src_std = src_set.scaler.scale_
                target_robust_std[zero_mask] = src_std[zero_mask]
                target_robust_std[target_robust_std < 1e-6] = 1.0

            self.logger.info(f"Computed Robust Median (First 5): {target_robust_mean[:5]}")
            self.logger.info(f"Computed Robust Std    (First 5): {target_robust_std[:5]}")

            forced_stats = {'mean': target_robust_mean, 'std': target_robust_std}

            reuse_df = pd.DataFrame(raw_features)
            reuse_df['label'] = raw_labels

            #重新加载目标域数据
            self.logger.info(f"Reloading Target Domain ({id_trg}) with ROBUST stats...")

            tgt_set, tgt_loader = self._get_data(
                flag='train',
                step=step_size,
                entity_id=id_trg,
                use_test_data=True,
                train_stats=forced_stats,
                pre_loaded_data=reuse_df
            )

            _, vali_loader = self._get_data(
                flag='val',
                entity_id=id_trg,
                use_test_data=True,
                train_stats=forced_stats,
                pre_loaded_data=reuse_df
            )

            # 保存 Target 统计量供 Test 阶段使用
            self.target_mean = target_robust_mean
            self.target_std = target_robust_std

            np.save(os.path.join(self.model_save_path, 'target_mean.npy'), target_robust_mean)
            np.save(os.path.join(self.model_save_path, 'target_std.npy'), target_robust_std)
            # 这里的 train_mean 也存为 target stats
            np.save(os.path.join(self.model_save_path, 'train_mean.npy'), target_robust_mean)
            np.save(os.path.join(self.model_save_path, 'train_std.npy'), target_robust_std)

        else:
            # 非 UDA
            src_set, train_loader = self._get_data(flag='train', step=1)
            self.train_mean = src_set.scaler.mean_
            self.train_std = src_set.scaler.scale_
            np.save(os.path.join(self.model_save_path, 'train_mean.npy'), self.train_mean)
            np.save(os.path.join(self.model_save_path, 'train_std.npy'), self.train_std)
            _, vali_loader = self._get_data(flag='val')
            train_steps = len(train_loader)


        # 通用训练循环

        model_optim = self._select_optimizer()
        early_stopping = EarlyStopping(patience=self.train_configs.patience, verbose=True, logger=self.logger)

        lambda_adv = 0.1
        lambda_rec_tgt = 1.0
        time_now = time.time()

        for epoch in range(self.train_configs.train_epochs):
            iter_count = 0
            train_loss_list = []
            self.model.train()
            epoch_time = time.time()

            if self.use_uda:
                iter_src = iter(src_loader)
                iter_tgt = iter(tgt_loader)
                len_src = len(src_loader)
                len_tgt = len(tgt_loader)
                train_steps = max(len_src, len_tgt)

                for i in range(train_steps):
                    iter_count += 1
                    # Source Batch
                    try:
                        batch_x_src, _ = next(iter_src)
                    except StopIteration:
                        iter_src = iter(src_loader)
                        batch_x_src, _ = next(iter_src)

                    # Target Batch
                    try:
                        batch_x_tgt, _ = next(iter_tgt)
                    except StopIteration:
                        iter_tgt = iter(tgt_loader)
                        batch_x_tgt, _ = next(iter_tgt)

                    model_optim.zero_grad()
                    batch_x_src = batch_x_src.float().to(self.device)
                    batch_x_tgt = batch_x_tgt.float().to(self.device)

                    #Source Forward
                    rec_loss_s, lat_loss_s, d_out_s = self.model(
                        batch_x_src, None, None, None, domain='source', update_flag=True)
                    loss_adv_s = F.binary_cross_entropy(d_out_s, torch.ones_like(d_out_s))

                    #Target Forward
                    rec_loss_t, lat_loss_t, d_out_t = self.model(
                        batch_x_tgt, None, None, None, domain='target', update_flag=False)
                    loss_adv_t = F.binary_cross_entropy(d_out_t, torch.zeros_like(d_out_t))

                    # Total Loss
                    loss_adv = (loss_adv_s + loss_adv_t) * 0.5
                    loss = (rec_loss_s + lambda_rec_tgt * rec_loss_t) + \
                           0.1 * lat_loss_s + \
                           lambda_adv * loss_adv

                    train_loss_list.append(loss.item())
                    loss.backward()
                    model_optim.step()

                    if (i + 1) % 100 == 0:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.train_configs.train_epochs - epoch) * train_steps - i)
                        self.logger.info(
                            f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f} (rec_s: {rec_loss_s:.4f}, rec_t: {rec_loss_t:.4f})")
                        iter_count = 0
                        time_now = time.time()

            else:
                # 非 UDA
                for i, (batch_x, batch_y) in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    ms_loss, q_latent_distance, _ = self.model(batch_x, None, None, None)
                    loss = ms_loss
                    train_loss_list.append(loss.item())
                    loss.backward()
                    model_optim.step()

            self.logger.info("Epoch: {} cost time: {:.2f}".format(epoch + 1, time.time() - epoch_time))
            train_loss_avg = np.average(train_loss_list)
            vali_loss = self.vali(vali_loader, use_uda=self.use_uda)

            history['train_loss'].append(train_loss_avg)
            history['vali_loss'].append(vali_loss)

            self.logger.info(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss_avg, vali_loss))

            early_stopping(vali_loss, self.model, self.model_save_path)

            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

            if self.train_configs.lradj != "1cycle":
                self._adjust_learning_rate(model_optim, epoch + 1, self.train_configs)

        pd.DataFrame(history).to_csv(os.path.join(self.model_save_path, 'loss_log.csv'), index_label='epoch')
        best_model_path = self.model_save_path + '/checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))


        if self.use_uda:
            self._save_features_for_exp4(tgt_loader)


    def _save_features_for_exp4(self, tgt_loader):
        self.logger.info("Extracting L_rec and L_unc for Feature Experiment 4 (2D Plot)...")
        # 必须开启 train 模式以激活 Dropout 和随机掩码，从而产生不确定性
        self.model.train()

        all_L_rec = []
        all_L_unc = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in tgt_loader:
                batch_x = batch_x.float().to(self.device)
                all_labels.append(batch_y.cpu().numpy())

                # 双次随机推断
                try:
                    score_1, _, _, _ = self.model.infer(batch_x, None, None, None)
                    score_2, _, _, _ = self.model.infer(batch_x, None, None, None)
                except ValueError:  # 兼容如果 infer 只有两个返回值的情况
                    score_1, _ = self.model.infer(batch_x, None, None, None)
                    score_2, _ = self.model.infer(batch_x, None, None, None)

                # 计算 L_rec (两次的平均误差) 和 L_unc (两次的绝对差异)
                rec = 0.5 * (score_1 + score_2)
                unc = torch.abs(score_1 - score_2)


                if rec.dim() > 1:
                    rec = rec.mean(dim=list(range(1, rec.dim())))
                    unc = unc.mean(dim=list(range(1, unc.dim())))

                all_L_rec.append(rec.cpu().numpy())
                all_L_unc.append(unc.cpu().numpy())

        # 拼接所有批次
        rec_array = np.concatenate(all_L_rec, axis=0)
        unc_array = np.concatenate(all_L_unc, axis=0)

        # 展平 labels 以防维度错误
        label_array = np.concatenate(all_labels, axis=0)
        label_array = label_array.reshape(-1)
        # 如果是窗口级的数据，取窗口内最大值作为该窗口的label
        if label_array.size > len(rec_array):
            label_array = label_array.reshape(len(rec_array), -1).max(axis=-1)

        # 保存到当前实验的 results 文件夹中
        np.save(os.path.join(self.rst_save_path, 'real_L_rec.npy'), rec_array)
        np.save(os.path.join(self.rst_save_path, 'real_L_unc.npy'), unc_array)
        np.save(os.path.join(self.rst_save_path, 'real_labels.npy'), label_array)

        self.logger.info(f"成功保存真实特征！路径: {self.rst_save_path}")


    def test(self, **args):
        # 统计量加载逻辑
        train_stats = None
        target_entity = getattr(self.args, 'id_trg', None)

        # 始终加载 'train' 前缀
        stats_prefix = 'train'

        try:
            mean_path = os.path.join(self.model_save_path, f'{stats_prefix}_mean.npy')
            std_path = os.path.join(self.model_save_path, f'{stats_prefix}_std.npy')

            if os.path.exists(mean_path) and os.path.exists(std_path):
                mean_val = np.load(mean_path)
                std_val = np.load(std_path)
                train_stats = {'mean': mean_val, 'std': std_val}
                self.logger.info(f"Loaded statistics ({stats_prefix}) for test set normalization.")
            else:
                self.logger.warning(
                    f"No {stats_prefix} statistics found. Test set will use its own statistics (Risk of Leakage!).")
        except Exception as e:
            self.logger.warning(f"Failed to load statistics: {e}")

        # 加载数据
        if self.use_uda and target_entity:
            # UDA 模式下，测试 Target 数据
            _, test_loader = self._get_data(flag='test', entity_id=target_entity, use_test_data=True,
                                            train_stats=train_stats)
            self.logger.info(f"Testing on Target Domain: {target_entity}")
        else:
            _, test_loader = self._get_data(flag='test', train_stats=train_stats)
            self.logger.info("Testing on Default Test Set")

        self.logger.info('loading model...')
        self.model.load_state_dict(torch.load(os.path.join(self.model_save_path, 'checkpoint.pth')), strict=False)
        self.logger.info('Model loaded.')

        self.model.eval()
        with torch.no_grad():
            attens_energy = []
            test_labels = []

            # 用于保存特色实验所需的原始序列和解耦误差
            amp_energies = []
            grad_energies = []
            orig_data_list = []

            for i, (batch_x, batch_y) in enumerate(test_loader):
                test_labels.append(batch_y)

                # 收集原始数据
                orig_data_list.append(batch_x.cpu().numpy())

                batch_x = batch_x.float().to(self.device)

                # 兼容性接收 infer 结果
                try:
                    score, q_latent_distance, amp_error, grad_error = self.model.infer(batch_x, None, None, None)
                    amp_energies.append(amp_error.detach().cpu().numpy())
                    grad_energies.append(grad_error.detach().cpu().numpy())
                except ValueError:
                    # 如果还没有修改 infer，则回退到原逻辑，只接收前两个
                    score, q_latent_distance = self.model.infer(batch_x, None, None, None)

                score = score.detach().cpu().numpy()
                attens_energy.append(score)

            attens_energy = np.concatenate(attens_energy, axis=0)
            attens_energy = attens_energy.reshape(-1, attens_energy.shape[-1])
            test_energy = np.array(attens_energy)

            test_labels = torch.cat(test_labels, dim=0).detach().cpu().numpy().reshape(-1)
            test_gt = test_labels.astype(int)

            if len(np.unique(test_gt)) > 2:
                print(f"[Warning] Detected multiclass labels {np.unique(test_gt)}. Binarizing for AD evaluation.")
                test_gt = (test_gt != 0).astype(int)

        np.save(self.rst_save_path + '/a_gt.npy', test_gt)
        np.save(self.rst_save_path + '/a_test_energy.npy', test_energy)

        # 保存双视角误差与原始序列
        if amp_energies and grad_energies:
            amp_arr = np.concatenate(amp_energies, axis=0)
            grad_arr = np.concatenate(grad_energies, axis=0)
            orig_arr = np.concatenate(orig_data_list, axis=0)

            # 展平处理：处理多通道或多维度输出的情况，对通道取平均并压平为序列形式
            if amp_arr.ndim > 2: amp_arr = amp_arr.reshape(-1, amp_arr.shape[-1])
            if grad_arr.ndim > 2: grad_arr = grad_arr.reshape(-1, grad_arr.shape[-1])
            if orig_arr.ndim > 2: orig_arr = orig_arr.reshape(-1, orig_arr.shape[-1])

            amp_out = np.mean(amp_arr, axis=-1) if amp_arr.ndim > 1 else amp_arr
            grad_out = np.mean(grad_arr, axis=-1) if grad_arr.ndim > 1 else grad_arr
            orig_out = np.mean(orig_arr, axis=-1) if orig_arr.ndim > 1 else orig_arr

            np.save(self.rst_save_path + '/amp_error.npy', amp_out)
            np.save(self.rst_save_path + '/grad_error.npy', grad_out)
            np.save(self.rst_save_path + '/original_data.npy', orig_out)

            self.logger.info(f"特色实验数据 (original_data, amp_error, grad_error) 已保存至: {self.rst_save_path}")


        test_energy = np.mean(test_energy, axis=-1)
        # test_energy = np.max(test_energy, axis=-1)

        from ts_ad_evaluation import Evaluator
        evaluator = Evaluator(test_gt, test_energy, self.rst_save_path)
        results = evaluator.evaluate(metrics=self.args.metrics)

        self._save_summary(results)

    def _save_summary(self, results):
        summary = {}
        for metric_name, df_result in results.items():
            if isinstance(df_result, pd.DataFrame) and not df_result.empty:
                metric_dict = df_result.iloc[0].to_dict()
                for k, v in metric_dict.items():
                    summary[f"{metric_name}_{k}"] = v

        json_path = os.path.join(self.rst_save_path, 'final_metrics.json')

        def convert(o):
            if isinstance(o, np.int64): return int(o)
            if isinstance(o, np.float32): return float(o)
            return o

        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=4, default=convert)
        self.logger.info(f"Local metrics saved to {json_path}")

        try:
            csv_row = summary.copy()
            csv_row['Source_Domain'] = getattr(self.args, 'id_src', 'None')
            csv_row['Target_Domain'] = getattr(self.args, 'id_trg', 'None')

            head_cols = ['Source_Domain', 'Target_Domain']
            priority_metric = ['best_f1_F1_best', 'best_f1_Macro_F1_best']
            other_cols = [k for k in csv_row.keys() if k not in head_cols and k not in priority_metric]
            cols_order = head_cols + priority_metric + other_cols

            df_new = pd.DataFrame([csv_row])
            df_new = df_new.reindex(columns=cols_order)

            dataset_dir = os.path.dirname(self.rst_save_path)
            common_csv_path = os.path.join(dataset_dir, f'{self.args.data}_summary.csv')
            file_exists = os.path.isfile(common_csv_path)

            df_new.to_csv(common_csv_path, mode='a', header=not file_exists, index=False)
            self.logger.info(f"Global metrics appended to {common_csv_path}")
        except Exception as e:
            self.logger.error(f"Failed to append to summary CSV: {e}")

    def evaluate_spot(self, **args):
        pass

    def analysis(self):
        self._basic()