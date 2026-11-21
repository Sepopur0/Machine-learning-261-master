"""Kiến trúc MLP, ResNet và FT-Transformer (cài đặt PyTorch thuần, theo ý tưởng của Gorishniy et al. 2021).

Lưu ý: đây là bản cài đặt đơn giản hóa để nhóm kiểm soát được từng dòng code. Nếu muốn dùng bản
tham chiếu, thay bằng `rtdl_revisiting_models` hoặc PyTorch Tabular (cần kiểm tra API hiện tại).
Mọi mô hình nhận (x_num, x_cat) và trả logits dạng (B, out_dim).
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoryEmbeddings(nn.Module):
    """Mỗi cột categorical có bảng embedding riêng (chỉ số 0 = giá trị lạ/hiếm)."""

    def __init__(self, cardinalities: List[int], dim: int):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, dim) for c in cardinalities])
        self.dim = dim

    def forward(self, x_cat):                      # (B, n_cat) -> (B, n_cat, dim)
        return torch.stack([e(x_cat[:, i]) for i, e in enumerate(self.embs)], dim=1)


class MLP(nn.Module):
    """MLP: [Linear -> ReLU -> Dropout] x n_layers -> Linear. Categorical qua embedding rồi nối với numerical."""

    def __init__(self, n_num, cat_cards, out_dim, d_hidden=128, n_layers=2, dropout=0.1, d_emb=8):
        super().__init__()
        self.cat = CategoryEmbeddings(cat_cards, d_emb) if cat_cards else None
        d_in = n_num + len(cat_cards) * d_emb
        layers, d = [], d_in
        for _ in range(n_layers):
            layers += [nn.Linear(d, d_hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = d_hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, out_dim)

    def forward(self, x_num, x_cat):
        parts = [x_num]
        if self.cat is not None:
            parts.append(self.cat(x_cat).flatten(1))
        return self.head(self.body(torch.cat(parts, dim=1)))


class ResBlock(nn.Module):
    def __init__(self, d_main, d_hidden, dropout_hidden, dropout_out):
        super().__init__()
        self.norm = nn.BatchNorm1d(d_main)
        self.lin1 = nn.Linear(d_main, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d_main)
        self.do_hidden = nn.Dropout(dropout_hidden)
        self.do_out = nn.Dropout(dropout_out)

    def forward(self, x):
        h = self.lin1(self.norm(x))
        h = self.do_hidden(F.relu(h))
        h = self.do_out(self.lin2(h))
        return x + h                                # kết nối tắt (residual)


class ResNet(nn.Module):
    """ResNet cho dữ liệu bảng: Linear -> [BN, Linear, ReLU, Dropout, Linear, Dropout, +skip] x n_blocks -> BN -> ReLU -> Linear."""

    def __init__(self, n_num, cat_cards, out_dim, d_main=128, d_hidden_factor=2.0, n_blocks=2,
                 dropout_hidden=0.1, dropout_out=0.05, d_emb=8):
        super().__init__()
        self.cat = CategoryEmbeddings(cat_cards, d_emb) if cat_cards else None
        d_in = n_num + len(cat_cards) * d_emb
        self.first = nn.Linear(d_in, d_main)
        d_hidden = max(1, int(d_main * d_hidden_factor))
        self.blocks = nn.Sequential(*[ResBlock(d_main, d_hidden, dropout_hidden, dropout_out)
                                      for _ in range(n_blocks)])
        self.last_norm = nn.BatchNorm1d(d_main)
        self.head = nn.Linear(d_main, out_dim)

    def forward(self, x_num, x_cat):
        parts = [x_num]
        if self.cat is not None:
            parts.append(self.cat(x_cat).flatten(1))
        h = self.blocks(self.first(torch.cat(parts, dim=1)))
        return self.head(F.relu(self.last_norm(h)))


class ReGLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


class FTBlock(nn.Module):
    """Khối Transformer pre-norm: LN -> MHA -> residual; LN -> FFN(ReGLU) -> residual."""

    def __init__(self, d_token, n_heads, attn_dropout, ffn_dropout, ffn_factor=4 / 3):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_token)
        d_ffn = int(d_token * ffn_factor)
        self.ffn = nn.Sequential(nn.Linear(d_token, d_ffn * 2), ReGLU(), nn.Dropout(ffn_dropout),
                                 nn.Linear(d_ffn, d_token))
        self.res_dropout = nn.Dropout(ffn_dropout)

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.res_dropout(h)
        x = x + self.res_dropout(self.ffn(self.norm2(x)))
        return x


class FTTransformer(nn.Module):
    """FT-Transformer: Feature Tokenizer (mỗi feature -> một token) + token [CLS] + các khối Transformer.

    - Numerical: token_i = x_i * W_i + b_i (mỗi feature có W_i, b_i riêng).
    - Categorical: token_j = Embedding_j(x_j) + b_j.
    - Dự đoán lấy từ token [CLS] sau LN -> ReLU -> Linear.
    """

    def __init__(self, n_num, cat_cards, out_dim, d_token=64, n_blocks=2, n_heads=4,
                 attn_dropout=0.1, ffn_dropout=0.1):
        super().__init__()
        assert d_token % n_heads == 0, "d_token phải chia hết cho n_heads"
        self.n_num = n_num
        bound = 1 / math.sqrt(d_token)
        if n_num:
            self.num_w = nn.Parameter(torch.empty(n_num, d_token).uniform_(-bound, bound))
            self.num_b = nn.Parameter(torch.empty(n_num, d_token).uniform_(-bound, bound))
        self.cat = CategoryEmbeddings(cat_cards, d_token) if cat_cards else None
        if cat_cards:
            self.cat_b = nn.Parameter(torch.empty(len(cat_cards), d_token).uniform_(-bound, bound))
        self.cls = nn.Parameter(torch.empty(1, 1, d_token).uniform_(-bound, bound))
        self.blocks = nn.ModuleList([FTBlock(d_token, n_heads, attn_dropout, ffn_dropout)
                                     for _ in range(n_blocks)])
        self.final_norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, out_dim)

    def forward(self, x_num, x_cat):
        tokens = []
        if self.n_num:
            tokens.append(x_num.unsqueeze(-1) * self.num_w + self.num_b)    # (B, n_num, d)
        if self.cat is not None:
            tokens.append(self.cat(x_cat) + self.cat_b)                      # (B, n_cat, d)
        x = torch.cat(tokens, dim=1)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.head(F.relu(self.final_norm(x[:, 0])))
