# tabdl: phần việc của người A (DL + TabPFN, dataset credit-g)

Bộ code này cài MLP, ResNet, FT-Transformer (PyTorch thuần) và wrapper TabPFN, theo giao diện chung của nhóm.

## Cài đặt

```bash
pip install openml scikit-learn pandas numpy optuna torch tabpfn
```

Chạy trên Colab: bật GPU (Runtime > Change runtime type > T4), rồi dùng `--device cuda`.

## Kiểm tra nhanh (không cần mạng, không cần TabPFN)

```bash
python tests/smoke_test.py
```

## Chạy thực nghiệm credit-g

```bash
# Đủ giao thức: 5-fold x 3 lần lặp = 15 lần chạy, cùng 20 trial tuning cho MLP/ResNet/FT-T
python scripts/run_experiment.py --dataset credit_g \
    --models mlp resnet ft_transformer tabpfn --n-trials 20 --device cuda

# Learning curve mức nhỏ (tái dùng siêu tham số đã tuning)
python scripts/run_experiment.py --dataset credit_g \
    --models mlp resnet ft_transformer tabpfn --n-train 100 200 500 --device cuda
```

Kết quả: `results/credit_g_dl.csv` (định dạng dài), siêu tham số tốt nhất: `results/best_params/`.
Script tự bỏ qua các lần chạy đã có trong CSV, nên chạy lại sau khi phiên Colab bị ngắt sẽ tiếp tục từ chỗ dở.

## Cấu trúc

| File | Nội dung |
|---|---|
| `tabdl/data.py` | Tải dataset OpenML, tạo và lưu split (thuộc phạm vi người C) |
| `tabdl/metrics.py` | Metric theo task (thuộc phạm vi người C) |
| `tabdl/preprocess.py` | Tiền xử lý fit trên train: median, standard/quantile, mã hóa categorical, gộp giá trị hiếm |
| `tabdl/torch_models.py` | Kiến trúc MLP, ResNet, FT-Transformer |
| `tabdl/models.py` | Vòng huấn luyện (AdamW, early stopping), wrapper TabPFN, `get_model` |
| `tabdl/tune.py` | Không gian tìm kiếm và hàm `tune` (Optuna) |
| `scripts/run_experiment.py` | Chạy toàn bộ giao thức trên một dataset |
| `tests/smoke_test.py` | Kiểm tra code trên dữ liệu tổng hợp |

## Quy ước chung với B và C

- `get_model(name, task, cat_cols, params, seed, device)` trả về đối tượng có `fit(X_tr, y_tr, X_val, y_val)`, `predict`, `predict_proba`.
- Cùng số trial `--n-trials` cho mọi mô hình có tuning; TabPFN dùng cấu hình mặc định (ngoại lệ, ghi rõ trong báo cáo).
- Tuning một lần trên split đầu rồi dùng lại cho các lần chạy còn lại (có thể hơi lạc quan cho các split sau; áp dụng đồng nhất cho mọi mô hình).
- CSV kết quả: `dataset, model, run_id, n_train, metric, value, train_time, infer_time, params`. Cột `n_train` là phần mở rộng so với hợp đồng ban đầu; người C cần thêm vào script thống kê (khi so sánh chỉ lấy các dòng cùng `n_train`).
- Cùng file split `splits/<dataset>.json` cho mọi mô hình, để các lần chạy ghép cặp được khi kiểm định thống kê.

## Những điểm cần kiểm tra

1. ID OpenML 46918 đúng là credit-g, số mẫu, số feature, tên các lớp (code đặt "bad" là lớp dương nếu nhãn là good/bad).
2. API của gói `tabpfn` (tên tham số `categorical_features_indices`, `n_estimators`, giới hạn mẫu, VRAM) có thể khác giữa v2 và 2.5; wrapper chưa được chạy thử vì cần tải trọng số.
3. Không gian tìm kiếm trong `tune.py` là thu nhỏ từ Gorishniy et al. (2021); đối chiếu lại với bài gốc.
4. Kiến trúc là bản đơn giản hóa; nếu muốn dùng bản tham chiếu, thay bằng `rtdl_revisiting_models` hoặc PyTorch Tabular.
5. `tests/smoke_test.py` đã chạy thành công (PyTorch 2.15, Optuna 5.0, CPU) trên dữ liệu tổng hợp cho cả 3 loại task. Chưa chạy trên dữ liệu thật credit-g (môi trường soạn code không truy cập được OpenML) và chưa chạy TabPFN (cần tải trọng số); cần chạy thử trên máy hoặc Colab của nhóm.
