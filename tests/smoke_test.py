"""Smoke test: kiểm tra code chạy được trên dữ liệu tổng hợp (không cần mạng, không cần TabPFN).

    python tests/smoke_test.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tabdl import Dataset, evaluate, get_model, make_splits, train_val_split, tune  # noqa: E402


def synthetic(task, n=400, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "a": rng.randn(n), "b": rng.randn(n), "c": rng.rand(n) * 10,
        "cat1": rng.choice(["x", "y", "z"], n), "cat2": rng.choice(list("abcdefgh"), n),
    })
    X.loc[rng.choice(n, 10, replace=False), "a"] = np.nan          # thử giá trị thiếu
    score = X["a"].fillna(0) + 0.5 * X["b"] + (X["cat1"] == "x") * 1.0 + rng.randn(n) * 0.5
    if task == "regression":
        y = score.to_numpy()
    elif task == "binary":
        y = (score > np.median(score)).astype(int).to_numpy()
    else:
        y = np.digitize(score, np.quantile(score, [1 / 3, 2 / 3])).astype(int)
    classes = None if task == "regression" else [str(i) for i in range(len(np.unique(y)))]
    return Dataset("synthetic_" + task, X, y, ["cat1", "cat2"], task, classes)


def main():
    for task in ["binary", "multiclass", "regression"]:
        ds = synthetic(task)
        splits = make_splits(ds, n_splits=3, n_repeats=1)
        tr, te = splits[0]
        tr_i, va_i = train_val_split(ds, tr, 0.2, seed=0)
        n_classes = None if task == "regression" else len(ds.classes)
        for name in ["mlp", "resnet", "ft_transformer"]:
            # tuning rất ngắn để kiểm tra luồng tune -> fit -> evaluate
            best = tune(name, task, ds.cat_cols, ds.X.iloc[tr_i], ds.y[tr_i], ds.X.iloc[va_i], ds.y[va_i],
                        n_trials=2, seed=0, n_classes=n_classes, fixed=dict(max_epochs=5))
            m = get_model(name, task, ds.cat_cols, best, seed=0, n_classes=n_classes)
            m.fit(ds.X.iloc[tr_i], ds.y[tr_i], ds.X.iloc[va_i], ds.y[va_i])
            Xte = ds.X.iloc[te]
            if task == "regression":
                res = evaluate(task, ds.y[te], m.predict(Xte))
            else:
                proba = m.predict_proba(Xte)
                assert proba.shape == (len(te), n_classes) and np.allclose(proba.sum(1), 1, atol=1e-4)
                res = evaluate(task, ds.y[te], proba.argmax(1), proba)
            print(f"OK  {task:10s} {name:15s} " + " ".join(f"{k}={v:.3f}" for k, v in res.items()))
    print("Tất cả smoke test đã chạy xong.")


if __name__ == "__main__":
    main()
