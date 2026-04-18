import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .EncDec import Transpose
import json


# 支持空洞卷积的残差块 (用于 TCN/CNN)
class DilatedResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.2):
        super(DilatedResBlock, self).__init__()

        # 动态计算 Padding 保持长度不变
        # 如果是标准 CNN (dilation=1)，padding 就是 (k-1)//2
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


# 辅助工具类
class MS_Utils(nn.Module):
    def __init__(self, kernels, method="interval_sampling"):
        super().__init__()
        self.kernels = kernels
        self.method = method

    def concat_sampling_list(self, x_enc_sampling_list):
        return torch.concat(x_enc_sampling_list, dim=1)

    def split_2_list(self, ms_x, ms_lens, mode="encoder"):
        if mode == "encoder":
            return list(torch.split(ms_x, split_size_or_sections=ms_lens[:-1], dim=1))
        elif mode == "decoder":
            return list(torch.split(ms_x, split_size_or_sections=ms_lens, dim=1))
        else:
            return list(torch.split(ms_x, split_size_or_sections=ms_lens, dim=1))

    def down(self, x_enc):
        x_enc_sampling_list = []
        for kernel in self.kernels:
            pad_x_enc = F.pad(x_enc, pad=(0, kernel - 1), mode="replicate")
            x_enc_i = pad_x_enc.unfold(dimension=-1, size=kernel, step=kernel)
            if self.method == "average_pooling":
                x_enc_i = torch.mean(x_enc_i, dim=-1)
            elif self.method == "interval_sampling":
                x_enc_i = x_enc_i[:, :, :, 0]
            x_enc_sampling_list.append(x_enc_i)
        return x_enc_sampling_list

    def up(self, x_enc_list, ms_lens):
        up_list = []
        for i in range(len(x_enc_list)):
            if i + 1 >= len(ms_lens): break
            curr_x = x_enc_list[i]
            target_len = ms_lens[i + 1]
            curr_x = curr_x.permute(0, 2, 1)
            up_x = F.interpolate(curr_x, size=target_len, mode='nearest')
            up_x = up_x.permute(0, 2, 1)
            up_list.append(up_x)
        return up_list

    @torch.no_grad()
    def _dummy_forward(self, input_len):
        dummy_x = torch.ones((1, 1, input_len))
        dummy_sampling_list = self.down(dummy_x)
        ms_t_lens = [s.shape[2] for s in dummy_sampling_list]
        ms_t_lens.append(input_len)
        return ms_t_lens

    def forward(self, x_enc):
        return self.down(x_enc)


# PatchEmbedding
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout, in_channels=1):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        self.value_embedding = nn.Linear(patch_len * in_channels, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def _dummy_forward(self, input_lens):
        ms_p_lens = []
        for input_len in input_lens:
            dummy_x = torch.ones((1, 1, input_len))
            dummy_x = self.padding_patch_layer(dummy_x)
            dummy_x = dummy_x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            ms_p_lens.append(dummy_x.shape[2])
        return ms_p_lens

    def forward(self, x):
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))
        x = self.value_embedding(x)
        return self.dropout(x), None


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, t_len):
        return self.pe[:, :t_len]


class Configs:
    def __init__(self, json_path):
        with open(json_path) as f:
            configs = json.load(f)
            self.__dict__.update(configs)


# AMCAD 主模型 (支持 TCN / CNN / Transformer)

class Basic_AMCAD(nn.Module):
    def __init__(self, configs):
        super(Basic_AMCAD, self).__init__()

        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len
        self.d_model = configs.d_model

        self.iqr_mult = getattr(configs, 'iqr_mult', 1.5)

        # 选择骨干网络类型
        # 默认使用 TCN (原版), 可选 'CNN' 或 'Transformer'
        self.backbone_type = getattr(configs, 'backbone_type', 'TCN')
        print(f"Build AMCAD with backbone: {self.backbone_type}")

        # 1. 多尺度生成
        self.ms_utils = MS_Utils(configs.ms_kernels, configs.ms_method)
        self.n_scales = len(configs.ms_kernels)

        # 2. Embedding
        self.patch_embedding = PatchEmbedding(configs.d_model, configs.patch_len,
                                              configs.patch_len, configs.patch_len - 1, 0.,
                                              in_channels=2)

        self.pos_embedding = PositionalEmbedding(configs.d_model)

        self.ms_t_lens = self.ms_utils._dummy_forward(configs.seq_len)
        self.ms_p_lens = self.patch_embedding._dummy_forward(self.ms_t_lens)

        self.mask_ratio = 0.2

        # 3. 骨干网络构建
        if self.backbone_type == 'Transformer':
            #Transformer 结构
            encoder_layer = nn.TransformerEncoderLayer(d_model=configs.d_model,
                                                       nhead=4,  # 可在 config 中调整
                                                       dim_feedforward=configs.d_model * 4,
                                                       dropout=0.2,
                                                       batch_first=True)
            self.encoder_trans = nn.TransformerEncoder(encoder_layer, num_layers=configs.e_layers)

            # Decoder 部分也使用 TransformerEncoder 结构堆叠
            decoder_layer = nn.TransformerEncoderLayer(d_model=configs.d_model,
                                                       nhead=4,
                                                       dim_feedforward=configs.d_model * 4,
                                                       dropout=0.2,
                                                       batch_first=True)
            self.decoder_trans = nn.TransformerEncoder(decoder_layer, num_layers=configs.d_layers)

        else:
            # TCN 或 标准CNN 结构
            # Encoder
            enc_layers = []
            for i in range(configs.e_layers):
                if self.backbone_type == 'CNN':
                    dilation_rate = 1  # 标准 CNN，无膨胀
                else:
                    dilation_rate = 2 ** i  # TCN，指数膨胀

                enc_layers.append(
                    DilatedResBlock(channels=configs.d_model, dilation=dilation_rate, dropout=0.2)
                )
            self.encoder_cnn = nn.Sequential(*enc_layers)

            # Decoder
            dec_layers = []
            for i in range(configs.d_layers):
                if self.backbone_type == 'CNN':
                    dilation_rate = 1
                else:
                    dilation_rate = 2 ** (configs.d_layers - 1 - i)

                dec_layers.append(
                    DilatedResBlock(channels=configs.d_model, dilation=dilation_rate, dropout=0.2)
                )
            self.decoder_cnn = nn.Sequential(*dec_layers)

        # 5. Projection
        self.projection = nn.Sequential(
            nn.Linear(configs.d_model, configs.patch_len * 2),
            nn.Flatten(-2)
        )

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.ones([N, L], device=x.device)
        mask[:, len_keep:] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore).unsqueeze(-1)
        return x * mask, mask

    def _forward(self, x_enc, domain='source', update_flag=True):
        bs, t, c = x_enc.shape

        x_diff = x_enc - torch.roll(x_enc, 1, dims=1)
        x_diff[:, 0, :] = 0

        x_val_in = x_enc.permute(0, 2, 1).reshape(bs * c, 1, t)
        x_diff_in = x_diff.permute(0, 2, 1).reshape(bs * c, 1, t)
        x_input = torch.cat([x_val_in, x_diff_in], dim=1)

        # 1. 多尺度下采样
        ms_x_list = self.ms_utils(x_input)


        ms_gt_list = ms_x_list
        ms_gt_cat = [s.permute(0, 2, 1) for s in ms_gt_list]
        ms_gt = self.ms_utils.concat_sampling_list(ms_gt_cat)

        # 2. Patch Embedding
        ms_emb_list = []
        for x_scale in ms_x_list:
            emb, _ = self.patch_embedding(x_scale)
            ms_emb_list.append(emb)
        ms_x_enc = self.ms_utils.concat_sampling_list(ms_emb_list)

        # Positional Embedding
        pos_emb = self.pos_embedding(self.ms_p_lens[-1])
        ms_pos_emb_list = self.ms_utils(pos_emb.permute(0, 2, 1))
        ms_pos_emb_list = [p.permute(0, 2, 1) for p in ms_pos_emb_list]
        ms_pos_emb = self.ms_utils.concat_sampling_list(ms_pos_emb_list)
        ms_x_enc = ms_x_enc + ms_pos_emb

        if self.training and update_flag:
            ms_x_enc, mask = self.random_masking(ms_x_enc, self.mask_ratio)

        # 3. 骨干网络前向传播
        if self.backbone_type == 'Transformer':
            # Transformer 输入格式: [Batch, SeqLen, Dim]
            x_encoded = self.encoder_trans(ms_x_enc)
            x_decoded = self.decoder_trans(x_encoded)
            ms_x_dec = x_decoded  # [B, N, D]
        else:
            # TCN / CNN 输入格式: [Batch, Dim, SeqLen]
            x_cnn_in = ms_x_enc.permute(0, 2, 1)
            x_encoded = self.encoder_cnn(x_cnn_in)
            x_decoded = self.decoder_cnn(x_encoded)
            ms_x_dec = x_decoded.permute(0, 2, 1)  # 变回 [B, N, D]

        if domain == 'source':
            domain_logit = torch.ones(bs, 1, device=x_enc.device)
        else:
            domain_logit = torch.zeros(bs, 1, device=x_enc.device)
        mem_loss = torch.tensor(0.0, device=x_enc.device)

        # 4. Output Projection
        ms_x_dec = self.projection(ms_x_dec)
        ms_x_dec = ms_x_dec.reshape(bs * c, -1, 2)

        gt_lens_t = self.ms_t_lens[:-1]
        gt_lens_p = self.ms_p_lens[:-1]

        ms_x_dec_list = list(
            torch.split(ms_x_dec, split_size_or_sections=[p * self.patch_len for p in gt_lens_p], dim=1))

        for i in range(len(ms_x_dec_list)):
            ms_x_dec_list[i] = ms_x_dec_list[i][:, :gt_lens_t[i], :]

        ms_x_dec = self.ms_utils.concat_sampling_list(ms_x_dec_list)

        return ms_gt, ms_x_dec, mem_loss, domain_logit

    def _ms_anomaly_score(self, ms_x_dec, ms_gt):
        #计算未平均的 MSE 损失，形状保持为 [bs*c, total_len, 2]
        ms_score = F.mse_loss(ms_x_dec, ms_gt, reduction="none")

        gt_lens_t = self.ms_t_lens[:-1]
        ms_score_list = list(torch.split(ms_score, split_size_or_sections=gt_lens_t, dim=1))
        orig_len = self.ms_t_lens[-1]

        final_score = torch.zeros(ms_score.shape[0], orig_len, ms_score.shape[2]).to(ms_score.device)
        for i in range(len(ms_score_list)):
            loss_i = ms_score_list[i].permute(0, 2, 1)
            up_loss_i = F.interpolate(loss_i, size=orig_len, mode='linear').permute(0, 2, 1)
            final_score = final_score + up_loss_i

        # 多尺度平均
        final_score = final_score / len(ms_score_list)  # 此时形状为 [bs*c, orig_len, 2]

        #解耦误差
        score_mean = final_score.mean(dim=-1)  # 综合异常分数
        score_amp = final_score[:, :, 0]  # 仅幅值重构误差
        score_grad = final_score[:, :, 1]  # 仅梯度重构误差

        return score_mean, score_amp, score_grad

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, domain='source', update_flag=True):

        ms_gt_1, ms_x_dec_1, mem_loss, domain_logit = self._forward(x_enc, domain, update_flag)

        if domain == 'target' and self.training:
            _, ms_x_dec_2, _, _ = self._forward(x_enc, domain, update_flag)
            raw_loss_1 = F.mse_loss(ms_x_dec_1, ms_gt_1, reduction='none').mean(dim=[1, 2])
            raw_loss_2 = F.mse_loss(ms_x_dec_2, ms_gt_1, reduction='none').mean(dim=[1, 2])
            recon_error = (raw_loss_1 + raw_loss_2) / 2
            uncertainty = F.mse_loss(ms_x_dec_1, ms_x_dec_2, reduction='none').mean(dim=[1, 2])
            final_score = recon_error + 1.0 * uncertainty

            q1 = torch.quantile(final_score, 0.25)
            q3 = torch.quantile(final_score, 0.75)
            iqr = q3 - q1

            base_threshold = q3 + 1.5 * iqr

            # 强行关闭筛选的后门逻辑：如果 iqr_mult 传入负数，则掩码全部设为 True（全盘接收污染）
            if self.iqr_mult < 0:
                mask = torch.ones_like(final_score, dtype=torch.bool)
            else:
                dynamic_threshold = self.iqr_mult * base_threshold
                mask = final_score < dynamic_threshold

            if mask.sum() > 0:
                loss_valid = recon_error[mask].mean() + 0.5 * uncertainty[mask].mean()
                recon_loss = loss_valid
            else:
                recon_loss = recon_error.mean()
        else:
            recon_loss = F.mse_loss(ms_x_dec_1, ms_gt_1)

        return recon_loss, mem_loss, domain_logit

    def infer(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        ms_gt, ms_x_dec, mem_loss, _ = self._forward(x_enc, domain='target', update_flag=False)

        # 接收解耦后的三个分数
        score, amp_error, grad_error = self._ms_anomaly_score(ms_x_dec, ms_gt)

        bs = x_enc.shape[0]
        c = x_enc.shape[2]

        # 将形状从 [bs*c, seq_len] 恢复为 [bs, seq_len, c]
        score = score.reshape(bs, c, -1).permute(0, 2, 1)
        amp_error = amp_error.reshape(bs, c, -1).permute(0, 2, 1)
        grad_error = grad_error.reshape(bs, c, -1).permute(0, 2, 1)

        return score, mem_loss, amp_error, grad_error