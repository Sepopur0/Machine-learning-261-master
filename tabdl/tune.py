"""Tuning bằng Optuna. Yêu cầu của đề: CÙNG ngân sách tuning cho mọi mô hình -> cùng n_trials.

Khoảng giá trị dựa trên hướng của Gorishniy et al. (2021) nhưng thu nhỏ cho dữ liệu nhỏ [cần kiểm tra
lại với bài gốc]. Kích thước không gian tìm kiếm của ba mô hình tương đương nhau (6-8 siêu tham số).
Mục tiêu tối thiểu hóa: log-loss trên validation (phân loại) hoặc RMSE (hồi quy).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import optuna
from sklearn.metrics import log_loss, mean_squared_error

from .models import get_model


def suggest(name: str, trial: "optuna.Trial") -> dict:
    if name == "mlp":
        return dict(
            n_layers=trial.suggest_int("n_layers", 1, 4),
            d_hidden=trial.suggest_categorical("d_hidden", [32, 64, 128, 256]),
            dropout=trial.suggest_float("dropout", 0.0, 0.5),
            d_emb=trial.suggest_categorical("d_emb", [4, 8, 16]),
            lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            num_prep=trial.suggest_categorical("num_prep", ["standard", "quantile"]),
        )
    if name == "resnet":
        return dict(
            n_blocks=trial.suggest_int("n_blocks", 1, 4),
            d_main=trial.suggest_categorical("d_main", [32, 64, 128, 256]),
            d_hidden_factor=trial.suggest_float("d_hidden_factor", 1.0, 4.0),
            dropout_hidden=trial.suggest_float("dropout_hidden", 0.0, 0.5),
            dropout_out=trial.suggest_float("dropout_out", 0.0, 0.3),
            lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            num_prep=trial.suggest_categorical("num_prep", ["standard", "quantile"]),
        )
    if name == "ft_transformer":
        return dict(
            n_blocks=trial.suggest_int("n_blocks", 1, 3),
            d_token=trial.suggest_categorical("d_token", [32, 64, 128]),
            attn_dropout=trial.suggest_float("attn_dropout", 0.0, 0.4),
            ffn_dropout=trial.suggest_float("ffn_dropout", 0.0, 0.4),
            lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            num_prep=trial.suggest_categorical("num_prep", ["standard", "quantile"]),
        )
    raise ValueError(f"Không có không gian tìm kiếm cho {name} (TabPFN không tuning).")


def tune(name: str, task: str, cat_cols, X_tr, y_tr, X_val, y_val, n_trials: int = 20, seed: int = 0,
         device: str = "cpu", n_classes: Optional[int] = None, timeout: Optional[float] = None,
         fixed: Optional[dict] = None) -> dict:
    """Trả về dict siêu tham số tốt nhất. `fixed`: tham số cố định thêm (ví dụ max_epochs để chạy thử)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {**suggest(name, trial), **(fixed or {})}
        m = get_model(name, task, cat_cols, params, seed=seed, device=device, n_classes=n_classes)
        m.fit(X_tr, y_tr, X_val, y_val)
        if task == "regression":
            return float(np.sqrt(mean_squared_error(y_val, m.predict(X_val))))
        proba = m.predict_proba(X_val)
        return float(log_loss(y_val, proba, labels=np.arange(proba.shape[1])))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return {**study.best_params, **(fixed or {})}
