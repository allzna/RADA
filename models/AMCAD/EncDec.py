import torch
import torch.nn as nn
import torch.nn.functional as F
from .ema import EMA


class Transpose(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, norm="batchnorm", dropout=0.1, activation="gelu"):
        super(EncoderLayer, self).__init__()
        self.ema = EMA(channels=d_model, factor=8)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)

        if "batch" in norm.lower():
            self.norm1 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
            self.norm2 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        x_in = x.permute(0, 2, 1).unsqueeze(-1)
        x_ema = self.ema(x_in)
        x_out = x_ema.squeeze(-1).permute(0, 2, 1)
        x = self.norm1(x + self.dropout(x_out))

        y = x.transpose(-1, 1)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y)).transpose(-1, 1)
        return self.norm2(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, cross_mask=None):
        for layer in self.layers:
            x, _, cross_attn_weights = layer(x, cross, cross_mask)

        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x, None, cross_attn_weights


class DecoderLayer(nn.Module):
    def __init__(self, cross_attention, d_model, d_ff, norm="batchnorm", dropout=0.1, activation="gelu"):
        super(DecoderLayer, self).__init__()
        self.self_ema = EMA(channels=d_model, factor=8)
        self.cross_attention = cross_attention

        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)

        if "batch" in norm.lower():
            self.norm1 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
            self.norm2 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
            self.norm3 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, cross_mask=None):
        x_in = x.permute(0, 2, 1).unsqueeze(-1)
        x_ema = self.self_ema(x_in)
        x_out = x_ema.squeeze(-1).permute(0, 2, 1)
        x = self.norm1(x + self.dropout(x_out))

        x_, cross_attn_weights = self.cross_attention(x, cross, cross, attn_mask=cross_mask)
        x = self.norm2(x + self.dropout(x_))

        y = x.transpose(-1, 1)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y)).transpose(-1, 1)
        return self.norm3(x + y), None, cross_attn_weights