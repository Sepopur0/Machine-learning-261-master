"""tabdl: module deep learning + TabPFN cho bài tập lớn (phần việc của người A).

Giao diện chung (hợp đồng với người B và C):
    load_dataset(name) -> Dataset(X, y, cat_cols, task, classes)
    make_splits(dataset, ...) -> danh sách (train_idx, test_idx), lưu vào splits/<dataset>.json
    get_model(name, task, cat_cols, params, seed, device) -> đối tượng có fit / predict / predict_proba
    tune(...) -> dict siêu tham số tốt nhất (cùng số trial cho mọi mô hình)
    evaluate(task, y_true, y_pred, proba) -> dict metric
"""
from .data import Dataset, load_dataset, make_splits, load_splits, train_val_split
from .metrics import evaluate, PRIMARY
from .models import get_model, MODEL_NAMES
from .tune import tune
