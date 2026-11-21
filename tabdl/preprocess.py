"""Tiền xử lý cho mô hình deep learning. Mọi thống kê được fit trên TRAIN rồi áp dụng cho val/test."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler


class TabPreprocessor:
    """Biến DataFrame thành (x_num float32, x_cat int64).

    - Numerical: điền thiếu bằng median (train), rồi standardize hoặc quantile (num_prep).
    - Categorical: mã hóa thành chỉ số 1..K theo train; 0 dành cho giá trị lạ/thiếu.
      Giá trị hiếm (tần suất < min_freq trong train) được gộp vào chỉ số 0 ("other").
    """

    def __init__(self, cat_cols: List[str], num_prep: str = "standard", min_freq: int = 1, seed: int = 0):
        self.cat_cols = list(cat_cols)
        self.num_prep = num_prep
        self.min_freq = min_freq
        self.seed = seed

    def fit(self, X: pd.DataFrame):
        self.num_cols = [c for c in X.columns if c not in self.cat_cols]
        self.medians_ = X[self.num_cols].median() if self.num_cols else None
        if self.num_cols:
            Xn = X[self.num_cols].fillna(self.medians_).to_numpy(dtype=np.float64)
            if self.num_prep == "quantile":
                n_q = max(10, min(1000, X.shape[0] // 5))
                self.scaler_ = QuantileTransformer(n_quantiles=n_q, output_distribution="normal",
                                                   subsample=10**9, random_state=self.seed)
            else:
                self.scaler_ = StandardScaler()
            self.scaler_.fit(Xn)
        self.cat_maps_ = {}
        for c in self.cat_cols:
            vc = X[c].astype(str).value_counts()
            keep = vc[vc >= self.min_freq].index.tolist()
            self.cat_maps_[c] = {v: i + 1 for i, v in enumerate(keep)}
        self.cat_cardinalities = [len(self.cat_maps_[c]) + 1 for c in self.cat_cols]
        self.n_num = len(self.num_cols)
        self.n_cat = len(self.cat_cols)
        return self

    def transform(self, X: pd.DataFrame):
        if self.n_num:
            Xn = X[self.num_cols].fillna(self.medians_).to_numpy(dtype=np.float64)
            x_num = self.scaler_.transform(Xn).astype(np.float32)
        else:
            x_num = np.zeros((len(X), 0), dtype=np.float32)
        if self.n_cat:
            cols = [X[c].astype(str).map(self.cat_maps_[c]).fillna(0).astype(np.int64).to_numpy()
                    for c in self.cat_cols]
            x_cat = np.stack(cols, axis=1)
        else:
            x_cat = np.zeros((len(X), 0), dtype=np.int64)
        return x_num, x_cat
