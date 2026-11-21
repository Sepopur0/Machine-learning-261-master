"""Chạy MLP / ResNet / FT-Transformer / TabPFN trên một dataset với giao thức chung của nhóm.

Ví dụ (credit-g, đủ giao thức: 5-fold x 3 lần lặp = 15 lần chạy, cùng 20 trial tuning):
    python scripts/run_experiment.py --dataset credit_g --models mlp resnet ft_transformer tabpfn \
        --n-trials 20 --n-splits 5 --n-repeats 3 --device cuda

Learning curve mức nhỏ (tái dùng siêu tham số đã tuning trên toàn bộ train):
    python scripts/run_experiment.py --dataset credit_g --models mlp ft_transformer tabpfn \
        --n-train 100 200 500 --device cuda

Kết quả ghi vào results/<dataset>_dl.csv theo định dạng dài (long format):
    dataset, model, run_id, n_train, metric, value, train_time, infer_time, params
(Cột n_train là phần mở rộng so với hợp đồng chung; người C cần thêm vào script thống kê.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tabdl import (MODEL_NAMES, evaluate, get_model, load_dataset, load_splits, make_splits,  # noqa: E402
                   train_val_split, tune)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="credit_g")
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    ap.add_argument("--n-trials", type=int, default=20, help="Cùng số trial cho mọi mô hình có tuning")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--n-train", type=int, nargs="*", default=None,
                    help="Learning curve: các cỡ tập train (subsample). Bỏ trống = dùng toàn bộ train.")
    ap.add_argument("--max-runs", type=int, default=None, help="Giới hạn số lần chạy (để chạy thử)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tpfn-max-train", type=int, default=10000,
                    help="Giới hạn mẫu train của TabPFN (v2 ~10k; bản 2.5 nhắm tới 50k) [kiểm tra]")
    return ap.parse_args()


def subsample(train_idx, y, n, seed, task):
    """Lấy n mẫu từ train (stratified nếu phân loại)."""
    if n >= len(train_idx):
        return train_idx
    rng = np.random.RandomState(seed)
    if task == "regression":
        return rng.choice(train_idx, n, replace=False)
    sel = []
    classes, counts = np.unique(y[train_idx], return_counts=True)
    for c, cnt in zip(classes, counts):
        pool = train_idx[y[train_idx] == c]
        k = max(2, int(round(n * cnt / len(train_idx))))
        sel.append(rng.choice(pool, min(k, len(pool)), replace=False))
    return np.concatenate(sel)


def main():
    a = parse()
    out = a.out or f"results/{a.dataset}_dl.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    os.makedirs("results/best_params", exist_ok=True)

    ds = load_dataset(a.dataset)
    print(f"[data] {ds.name}: X={ds.X.shape}, task={ds.task}, n_cat={len(ds.cat_cols)}")
    split_path = f"splits/{a.dataset}.json"
    if os.path.exists(split_path):
        splits = load_splits(split_path)
    else:
        splits = make_splits(ds, a.n_splits, a.n_repeats, seed=0, path=split_path)
    if a.max_runs:
        splits = splits[:a.max_runs]
    n_classes = None if ds.task == "regression" else len(ds.classes)

    done = set()
    if os.path.exists(out):
        prev = pd.read_csv(out)
        done = set(zip(prev["model"], prev["run_id"], prev["n_train"]))

    # --- 1. Tuning một lần trên split đầu (cùng n_trials cho mọi mô hình) ---
    tr0, _ = splits[0]
    tr_i, va_i = train_val_split(ds, tr0, 0.2, seed=0)
    best = {}
    for name in a.models:
        if name == "tabpfn":
            best[name] = {"max_train": a.tpfn_max_train}
            continue
        pfile = f"results/best_params/{a.dataset}_{name}.json"
        if os.path.exists(pfile):
            best[name] = json.load(open(pfile))
            print(f"[tune] {name}: dùng lại {pfile}")
            continue
        t0 = time.time()
        best[name] = tune(name, ds.task, ds.cat_cols, ds.X.iloc[tr_i], ds.y[tr_i], ds.X.iloc[va_i],
                          ds.y[va_i], n_trials=a.n_trials, seed=0, device=a.device, n_classes=n_classes)
        json.dump(best[name], open(pfile, "w"), indent=2)
        print(f"[tune] {name}: {time.time() - t0:.0f}s, best={best[name]}")

    # --- 2. Chạy trên mọi split ---
    sizes = a.n_train if a.n_train else [None]
    rows = []
    for run_id, (tr, te) in enumerate(splits):
        for n_tr in sizes:
            tr_use = tr if n_tr is None else subsample(tr, ds.y, n_tr, seed=run_id, task=ds.task)
            n_tr_label = len(tr_use)
            tr_i, va_i = train_val_split(ds, tr_use, 0.2, seed=run_id)
            for name in a.models:
                if (name, run_id, n_tr_label) in done:
                    continue
                m = get_model(name, ds.task, ds.cat_cols, best[name], seed=run_id, device=a.device,
                              n_classes=n_classes)
                t0 = time.time()
                m.fit(ds.X.iloc[tr_i], ds.y[tr_i], ds.X.iloc[va_i], ds.y[va_i])
                t_fit = time.time() - t0
                t0 = time.time()
                Xte = ds.X.iloc[te]
                proba = None if ds.task == "regression" else m.predict_proba(Xte)
                pred = m.predict(Xte) if ds.task == "regression" else proba.argmax(axis=1)
                t_inf = time.time() - t0
                metrics = evaluate(ds.task, ds.y[te], pred, proba)
                for k, v in metrics.items():
                    rows.append(dict(dataset=ds.name, model=name, run_id=run_id, n_train=n_tr_label,
                                     metric=k, value=v, train_time=t_fit, infer_time=t_inf,
                                     params=json.dumps(best[name])))
                main_metric = next(iter(metrics))
                print(f"[run {run_id:02d}] {name:15s} n_train={n_tr_label:5d} "
                      + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()) + f" fit={t_fit:.1f}s")
        # ghi dần để không mất kết quả nếu phiên bị ngắt
        if rows:
            pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            rows = []
    print(f"[done] kết quả: {out}")


if __name__ == "__main__":
    main()
