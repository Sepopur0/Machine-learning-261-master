"""Giao diện thống nhất cho các mô hình DL và TabPFN.

Mọi mô hình: fit(X_tr, y_tr, X_val, y_val), predict(X), predict_proba(X) (chỉ phân loại).
X là pandas DataFrame; cột categorical khai báo qua cat_cols.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import TabPreprocessor
from .torch_models import MLP, FTTransformer, ResNet

MODEL_NAMES = ["mlp", "resnet", "ft_transformer", "tabpfn"]

# Siêu tham số mặc định (dùng khi chưa tuning hoặc để chạy thử)
DEFAULTS = {
    "mlp": dict(d_hidden=128, n_layers=2, dropout=0.1, d_emb=8, lr=1e-3, weight_decay=1e-5),
    "resnet": dict(d_main=128, d_hidden_factor=2.0, n_blocks=2, dropout_hidden=0.1, dropout_out=0.05,
                   d_emb=8, lr=1e-3, weight_decay=1e-5),
    "ft_transformer": dict(d_token=64, n_blocks=2, n_heads=4, attn_dropout=0.1, ffn_dropout=0.1,
                           lr=1e-4, weight_decay=1e-5),
}
TRAIN_DEFAULTS = dict(batch_size=128, max_epochs=200, patience=16, num_prep="standard", min_freq=1)


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TorchTabModel:
    """Huấn luyện MLP / ResNet / FT-Transformer với early stopping trên validation."""

    def __init__(self, name: str, task: str, cat_cols, params: Optional[dict] = None,
                 seed: int = 0, device: str = "cpu", n_classes: Optional[int] = None):
        assert name in DEFAULTS, f"Mô hình không hợp lệ: {name}"
        self.name, self.task, self.cat_cols = name, task, list(cat_cols)
        self.params = {**DEFAULTS[name], **TRAIN_DEFAULTS, **(params or {})}
        self.seed, self.device, self.n_classes = seed, device, n_classes

    # ---------- nội bộ ----------
    def _out_dim(self):
        return self.n_classes if self.task == "multiclass" else 1

    def _build(self):
        p = self.params
        n_num, cards, out = self.prep.n_num, self.prep.cat_cardinalities, self._out_dim()
        if self.name == "mlp":
            return MLP(n_num, cards, out, p["d_hidden"], p["n_layers"], p["dropout"], p["d_emb"])
        if self.name == "resnet":
            return ResNet(n_num, cards, out, p["d_main"], p["d_hidden_factor"], p["n_blocks"],
                          p["dropout_hidden"], p["dropout_out"], p["d_emb"])
        return FTTransformer(n_num, cards, out, p["d_token"], p["n_blocks"], p["n_heads"],
                             p["attn_dropout"], p["ffn_dropout"])

    def _loss(self, logits, y):
        if self.task == "binary":
            return F.binary_cross_entropy_with_logits(logits.squeeze(-1), y.float())
        if self.task == "multiclass":
            return F.cross_entropy(logits, y)
        return F.mse_loss(logits.squeeze(-1), y)

    def _tensors(self, X, y=None):
        x_num, x_cat = self.prep.transform(X)
        out = [torch.from_numpy(x_num), torch.from_numpy(x_cat)]
        if y is not None:
            if self.task == "regression":
                y = (np.asarray(y, dtype=np.float32) - self.y_mean) / self.y_std
                out.append(torch.from_numpy(y.astype(np.float32)))
            else:
                out.append(torch.from_numpy(np.asarray(y, dtype=np.int64)))
        return out

    @torch.no_grad()
    def _forward_all(self, x_num, x_cat, bs=1024):
        self.net.eval()
        outs = []
        for i in range(0, len(x_num), bs):
            outs.append(self.net(x_num[i:i + bs].to(self.device), x_cat[i:i + bs].to(self.device)).float().cpu())
        return torch.cat(outs)

    # ---------- API ----------
    def fit(self, X_tr, y_tr, X_val, y_val):
        p = self.params
        _set_seed(self.seed)
        if self.task != "regression" and self.n_classes is None:
            self.n_classes = int(max(np.max(y_tr), np.max(y_val)) + 1)
        self.y_mean, self.y_std = 0.0, 1.0
        if self.task == "regression":
            self.y_mean, self.y_std = float(np.mean(y_tr)), float(np.std(y_tr) + 1e-8)
        self.prep = TabPreprocessor(self.cat_cols, p["num_prep"], p["min_freq"], self.seed).fit(X_tr)
        xn_tr, xc_tr, y_t = self._tensors(X_tr, y_tr)
        xn_va, xc_va, y_v = self._tensors(X_val, y_val)

        self.net = self._build().to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        n, bs = len(y_t), p["batch_size"]
        drop_last = (n % bs == 1)                       # tránh batch 1 mẫu (BatchNorm)
        gen = torch.Generator().manual_seed(self.seed)
        best, best_state, bad = float("inf"), None, 0

        for _ in range(p["max_epochs"]):
            self.net.train()
            perm = torch.randperm(n, generator=gen)
            end = n - (n % bs) if drop_last else n
            for i in range(0, end, bs):
                idx = perm[i:i + bs]
                opt.zero_grad(set_to_none=True)
                logits = self.net(xn_tr[idx].to(self.device), xc_tr[idx].to(self.device))
                loss = self._loss(logits, y_t[idx].to(self.device))
                loss.backward()
                opt.step()
            val_loss = float(self._loss(self._forward_all(xn_va, xc_va), y_v))
            if val_loss < best - 1e-6:
                best, best_state, bad = val_loss, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= p["patience"]:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.best_val_loss_ = best
        return self

    def predict_proba(self, X):
        assert self.task != "regression"
        logits = self._forward_all(*self._tensors(X))
        if self.task == "binary":
            p1 = torch.sigmoid(logits.squeeze(-1)).numpy()
            return np.stack([1 - p1, p1], axis=1)
        return torch.softmax(logits, dim=-1).numpy()

    def predict(self, X):
        if self.task == "regression":
            out = self._forward_all(*self._tensors(X)).squeeze(-1).numpy()
            return out * self.y_std + self.y_mean
        return self.predict_proba(X).argmax(axis=1)


class TabPFNModel:
    """Wrapper TabPFN. Không cần tuning; X_val không dùng (chỉ để giữ chung giao diện).

    Lưu ý [cần kiểm tra]: tên tham số của gói `tabpfn` có thể khác giữa các phiên bản (v2 / 2.5).
    Nếu số mẫu train vượt max_train (giới hạn của phiên bản đang dùng), sẽ lấy mẫu ngẫu nhiên.
    """

    def __init__(self, task: str, cat_cols, params: Optional[dict] = None, seed: int = 0,
                 device: str = "cpu", **_):
        self.task, self.cat_cols, self.seed, self.device = task, list(cat_cols), seed, device
        self.params = params or {}
        self.max_train = self.params.get("max_train", 10000)     # v2: ~10k; bản 2.5 nhắm tới 50k [?]
        self.n_estimators = self.params.get("n_estimators", None)

    def _encode(self, X, fit=False):
        from sklearn.preprocessing import OrdinalEncoder
        num_cols = [c for c in X.columns if c not in self.cat_cols]
        if fit:
            self.col_order_ = num_cols + self.cat_cols
            self.enc_ = (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
                         .fit(X[self.cat_cols].astype(str)) if self.cat_cols else None)
        parts = [X[num_cols].to_numpy(dtype=np.float64)]
        if self.cat_cols:
            parts.append(self.enc_.transform(X[self.cat_cols].astype(str)))
        return np.concatenate(parts, axis=1)

    def fit(self, X_tr, y_tr, X_val=None, y_val=None):
        _set_seed(self.seed)
        Xe = self._encode(X_tr, fit=True)
        y = np.asarray(y_tr)
        if len(y) > self.max_train:
            rng = np.random.RandomState(self.seed)
            sel = rng.choice(len(y), self.max_train, replace=False)
            Xe, y = Xe[sel], y[sel]
        n_num = Xe.shape[1] - len(self.cat_cols)
        cat_idx = list(range(n_num, Xe.shape[1]))
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        kw = dict(device=self.device, random_state=self.seed)
        if self.n_estimators:
            kw["n_estimators"] = self.n_estimators
        cls = TabPFNRegressor if self.task == "regression" else TabPFNClassifier
        try:
            self.model_ = cls(categorical_features_indices=cat_idx, **kw)
        except TypeError:
            self.model_ = cls(**kw)
        self.model_.fit(Xe, y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self._encode(X))

    def predict(self, X):
        Xe = self._encode(X)
        return self.model_.predict(Xe)


def get_model(name: str, task: str, cat_cols, params: Optional[dict] = None, seed: int = 0,
              device: str = "cpu", n_classes: Optional[int] = None):
    """Tạo mô hình theo tên: mlp | resnet | ft_transformer | tabpfn."""
    if name == "tabpfn":
        return TabPFNModel(task, cat_cols, params, seed, device)
    return TorchTabModel(name, task, cat_cols, params, seed, device, n_classes)
