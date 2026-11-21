"""Metric theo task. (Thuộc phạm vi người C; bản tối thiểu.)"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)


def evaluate(task: str, y_true, y_pred, proba=None) -> dict:
    """y_pred: nhãn (phân loại) hoặc giá trị (hồi quy). proba: ma trận xác suất (n, K) cho phân loại."""
    y_true = np.asarray(y_true)
    if task == "regression":
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }
    out = {"accuracy": float(accuracy_score(y_true, y_pred))}
    labels = np.arange(proba.shape[1])
    out["log_loss"] = float(log_loss(y_true, proba, labels=labels))
    if task == "binary":
        out["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
        out["f1"] = float(f1_score(y_true, y_pred))
    else:
        out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
        try:
            out["roc_auc_ovr"] = float(roc_auc_score(y_true, proba, multi_class="ovr", labels=labels))
        except ValueError:
            pass
    return out


# Metric chính theo task (dùng để so sánh)
PRIMARY = {"binary": "roc_auc", "multiclass": "macro_f1", "regression": "rmse"}
