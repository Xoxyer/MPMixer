import torch
from einops import rearrange
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from layers.PatchTST_layers import *
from typing import Callable, Optional
from layers.RevIN import RevIN
from layers.PatchTST_backbone import _MultiheadAttention
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
import math
import seaborn as sns

class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=True, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]
        self.res_attention = True
        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights



class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=True, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class MPMixerLayer(nn.Module):
    def __init__(self, configs, patch_size, patch_num, trans_d_model, dim, d_model, a, n_heads, d_k=None, d_v=None,res_attention=True, attn_dropout=0., proj_dropout=0.,kernel_size = 8):
        super().__init__()
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout,
                                             proj_dropout=proj_dropout, res_attention=res_attention)
        dropout = 0.05
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(d_model)

        self.num_down_sample = configs.num_down
        self.Resnet = nn.ModuleList([])
        for i in range(self.num_down_sample + 1):
            self.resnet = nn.Sequential(
                nn.Conv1d(dim[i], dim[i], kernel_size=kernel_size, groups=dim[i], padding='same'),
                nn.GELU(),
                nn.BatchNorm1d(dim[i])
            )
            self.Resnet.append(self.resnet)
        self.resnet_1 = nn.Sequential(
            nn.Conv1d(sum(dim), sum(dim), kernel_size=kernel_size, groups=sum(dim), padding='same'),
            nn.GELU(),
            nn.BatchNorm1d(sum(dim))
        )


    def forward(self, x):
        src = x + self.resnet_1(x)
        prev = None
        key_padding_mask = None
        attn_mask = None
        src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        src = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
        src = self.norm_attn(src)


        return src, attn, scores


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = Backbone(configs)
    def forward(self, x):
        x, attn, score = self.model(x)
        return x, attn, score
class Backbone(nn.Module):
    def __init__(self, configs, revin=True, affine=True, subtract_last=False):
        super().__init__()

        self.nvals = configs.enc_in
        self.lookback = configs.seq_len
        self.forecasting = configs.pred_len
        self.patch_size = configs.patch_len
        self.stride = configs.stride
        self.kernel_size = configs.mixer_kernel_size
        self.args = configs
        self.dropout = configs.dropout
        self.head_dropout = configs.head_dropout
        self.dropout = nn.Dropout(self.dropout)
        self.MPMixer_blocks = nn.ModuleList([])
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.trans_d_model = configs.trans_d_model
        self.depth = configs.tra_layers
        self.down_sampling_window = 2
        self.down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        self.patch_num = int((self.lookback - self.patch_size)/self.stride + 1) + 1
        self.num_down_sample = configs.num_down
        self.patch_length = []
        for i in range(self.num_down_sample + 1):
            self.patch_length.append(int(((self.lookback / (2**i)) - self.patch_size)/self.stride + 1) + 1)
        for _ in range(self.depth):
            self.MPMixer_blocks.append(MPMixerLayer(configs, patch_size=self.patch_size, patch_num=self.patch_num, trans_d_model=self.trans_d_model, d_model=self.d_model, dim=self.patch_length, a=self.patch_length[0], n_heads=configs.n_heads,kernel_size=self.kernel_size))

        patch_num_sum = sum(self.patch_length)
        self.head1 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear((patch_num_sum) * self.d_model, int(self.forecasting * 2)),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(int(self.forecasting * 2), self.forecasting),
            nn.Dropout(self.head_dropout)
        )
        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(self.nvals, affine=affine, subtract_last=subtract_last)


        self.Trend = nn.ModuleList([])
        self.Seasonal = nn.ModuleList([])
        self.W_P = nn.ModuleList([])
        for i in range(self.num_down_sample + 1):
            self.W_P.append(nn.Linear(self.patch_size, self.d_model))
            self.trend_linear = nn.Sequential(
                nn.Flatten(start_dim=-2),
                nn.Linear(self.patch_length[i] * self.d_model, self.forecasting),
                nn.Dropout(self.head_dropout)
            )
            self.Trend.append(self.trend_linear)
            self.seasonal_linear = nn.Sequential(
                nn.Flatten(start_dim=-2),
                nn.Linear(self.patch_length[i] * self.d_model, self.forecasting),
                nn.Dropout(self.head_dropout)
            )
            self.Seasonal.append(self.seasonal_linear)

        self.head3 = nn.Sequential(
            nn.Linear(self.forecasting, self.forecasting)
        )

        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)

    def forward(self, x):
        bs = x.shape[0]
        nvars = x.shape[-1]
        if self.revin:
            x = self.revin_layer(x, 'norm')
        x = x.permute(0, 2, 1)                                                       # x: [batch, n_val, seq_len]

        x_down = x
        input_series = []
        input_series.append(x)
        LIE_input_series = []
        LIE_input_series.append(x)
        for i in range(self.num_down_sample):
            x_down = self.down_pool(x_down)
            input_series.append(x_down)
            LIE_input_series.append(x_down)

        for i in range(self.num_down_sample + 1):
            x_lookback = self.padding_patch_layer(input_series[i])
            input_series[i] = x_lookback.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # x: [batch, n_val, patch_num, patch_size]
            input_series[i] = self.W_P[i](input_series[i])                                                              # x: [batch, n_val, patch_num, d_model]
            input_series[i] = torch.reshape(input_series[i], (input_series[i].shape[0] * input_series[i].shape[1], input_series[i].shape[2], input_series[i].shape[3]))      # x: [batch * n_val, patch_num, d_model]
            input_series[i] = self.dropout(input_series[i])

        trend_series = []
        seasonal_series = []

        for i in range(self.num_down_sample + 1):
            LIE_input_series[i] = LIE_input_series[i].permute(0, 2, 1)
            seasonal, trend = self.decompsition(LIE_input_series[i])
            seasonal_series.append(seasonal)
            trend_series.append(trend)

        i = 0
        for i in range(self.num_down_sample + 1):
            trend_series[i] = trend_series[i].permute(0, 2, 1)
            trend_lookback = self.padding_patch_layer(trend_series[i])
            trend_series[i] = trend_lookback.unfold(dimension=-1, size=self.patch_size,
                                                step=self.stride)  # x: [batch, n_val, patch_num, patch_size]
            trend_series[i] = self.W_P[i](trend_series[i])  # x: [batch, n_val, patch_num, d_model]
            trend_series[i] = torch.reshape(trend_series[i], (
            trend_series[i].shape[0] * trend_series[i].shape[1], trend_series[i].shape[2],
            trend_series[i].shape[3]))  # x: [batch * n_val, patch_num, d_model]
            trend_series[i] = self.dropout(trend_series[i])

        i = 0
        for i in range(self.num_down_sample + 1):
            seasonal_series[i] = seasonal_series[i].permute(0, 2, 1)
            seasonal_lookback = self.padding_patch_layer(seasonal_series[i])
            seasonal_series[i] = seasonal_lookback.unfold(dimension=-1, size=self.patch_size,
                                                    step=self.stride)  # x: [batch, n_val, patch_num, patch_size]
            seasonal_series[i] = self.W_P[i](seasonal_series[i])  # x: [batch, n_val, patch_num, d_model]
            seasonal_series[i] = torch.reshape(seasonal_series[i], (
                seasonal_series[i].shape[0] * seasonal_series[i].shape[1], seasonal_series[i].shape[2],
                seasonal_series[i].shape[3]))  # x: [batch * n_val, patch_num, d_model]
            seasonal_series[i] = self.dropout(seasonal_series[i])


        # Trend model
        sea = 0
        tre = 0
        i = 0
        j = 0
        # flag = 1
        # if flag == 0:
        #     tre = self.Trend[0](trend_series[0])
        #     sea = self.Seasonal[0](trend_series[0])

        for trend in self.Trend:
            tre += trend(trend_series[i])
            i += 1
        for seasonal in self.Seasonal:
            sea += seasonal(seasonal_series[j])
            j += 1

        # Period model
        src3 = input_series[0]
        for i in range(1, len(input_series)):
            src3 = torch.cat((src3, input_series[i]), dim=-2)

        for MPMixer_block in self.MPMixer_blocks:
            src3, attn, score = MPMixer_block(src3)

        res = src3
        res = self.head1(res)

        res = tre + sea + res

        res = torch.reshape(res, (bs, nvars, -1))  # res: [batch, n_val, pred_len]
        res = res.permute(0, 2, 1)


        if self.revin:
            res = self.revin_layer(res, 'denorm')
        return res, attn, score
