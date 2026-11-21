"""Tải dataset và tạo split.

Lưu ý: module này thuộc phạm vi người C; đây là bản tối thiểu để người A chạy được.
Khi C hoàn thiện bản chính thức, chỉ cần giữ nguyên chữ ký hàm.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedShuffleSplit

# Tên dataset -> ID OpenML (theo các dataset nhóm đã chọn)
OPENML_IDS = {
    "credit_g": 46918,
    "students": 46960,
    "house_sales": 44051,
}


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray            # nhãn đã mã hóa (0..K-1) hoặc giá trị thực (hồi quy)
    cat_cols: List[str]
    task: str                # "binary" | "multiclass" | "regression"
    classes: Optional[List[str]] = None


def load_dataset(name: str, cache_dir: str = "data_cache") -> Dataset:
    """Tải dataset từ OpenML (cần `pip install openml`). Ví dụ: load_dataset("credit_g")."""
    import openml

    os.makedirs(cache_dir, exist_ok=True)
    openml.config.set_root_cache_directory(os.path.abspath(cache_dir))
    d = openml.datasets.get_dataset(OPENML_IDS[name])
    X, y, cat_mask, names = d.get_data(target=d.default_target_attribute, dataset_format="dataframe")
    cat_cols = [n for n, is_cat in zip(names, cat_mask) if is_cat]
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype(str)

    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        task, classes, y_arr = "regression", None, y.to_numpy(dtype=float)
    else:
        labels = sorted(pd.Series(y).astype(str).unique())
        # Với credit-g, đặt lớp "bad" (rủi ro) là lớp dương = 1.
        if set(labels) == {"good", "bad"}:
            labels = ["good", "bad"]
        mapping = {lab: i for i, lab in enumerate(labels)}
        y_arr = pd.Series(y).astype(str).map(mapping).to_numpy(dtype=int)
        task = "binary" if len(labels) == 2 else "multiclass"
        classes = labels
    return Dataset(name=name, X=X.reset_index(drop=True), y=y_arr, cat_cols=cat_cols, task=task, classes=classes)


def make_splits(ds: Dataset, n_splits: int = 5, n_repeats: int = 3, seed: int = 0,
                path: Optional[str] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Repeated (stratified) K-fold. Lưu chỉ số vào file JSON để mọi người dùng chung một bộ split."""
    idx = np.arange(len(ds.y))
    if ds.task == "regression":
        splits = []
        for r in range(n_repeats):
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
            splits += [(tr, te) for tr, te in kf.split(idx)]
    else:
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        splits = [(tr, te) for tr, te in rskf.split(idx, ds.y)]
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump([{"train": tr.tolist(), "test": te.tolist()} for tr, te in splits], f)
    return splits


def load_splits(path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    with open(path) as f:
        raw = json.load(f)
    return [(np.array(s["train"]), np.array(s["test"])) for s in raw]


def train_val_split(ds: Dataset, train_idx: np.ndarray, val_frac: float = 0.2, seed: int = 0):
    """Tách validation từ phần train (stratified nếu là phân loại)."""
    y = ds.y[train_idx]
    if ds.task == "regression":
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(train_idx))
        n_val = int(len(train_idx) * val_frac)
        return train_idx[perm[n_val:]], train_idx[perm[:n_val]]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr, va = next(sss.split(np.zeros(len(y)), y))
    return train_idx[tr], train_idx[va]
