## Tổng quan dự án

Đây là dự án dự báo chuỗi thời gian khoảng (interval-valued time series forecasting) thuộc khóa luận tốt nghiệp (KLTN) triển khai các mô hình deep learning để dự đoán dữ liệu chuỗi thời gian với giá trị khoảng. Dự án tập trung vào các tác vụ dự báo sử dụng các mô hình như Transformer, LSTM, GRU và DLinear.

## Thiết lập môi trường

### Yêu cầu hệ thống
- **Python version**: 3.10.14 (khuyến nghị)
- **CUDA**: 12.8+ với cuDNN (cho GPU acceleration)
- **RAM**: Tối thiểu 8GB, khuyến nghị 16GB+
- **Disk**: Ít nhất 5GB trống

### Môi trường Python (Linux)
```bash
# Kiểm tra version Python
python3 --version  # Đảm bảo là 3.10.14

# Thiết lập môi trường tự động
bash builtEnv.sh
source .venv/bin/activate

# Kích hoạt nhanh (sau khi đã setup)
source ~/.bashrc
activate_env
```

### Môi trường Python (Windows)

#### Cách 1: Sử dụng PowerShell/Command Prompt
```powershell
# Kiểm tra version Python
python --version  # Đảm bảo là 3.10.14

# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo
.venv\Scripts\activate

# Cài đặt dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

#### Cách 2: Sử dụng Anaconda/Miniconda (Khuyến nghị)
```bash
# Tạo conda environment với Python 3.10.14
conda create -n kltn_env python=3.10.14

# Kích hoạt environment
conda activate kltn_env

# Cài đặt packages
pip install -r requirements.txt

# Hoặc cài từng package quan trọng
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install optuna pandas matplotlib seaborn scikit-learn
```

### Chạy code trên Windows

#### Huấn luyện mô hình:
```powershell
# Di chuyển vào thư mục MODELS
cd MODELS

# Kích hoạt environment (nếu dùng venv)
..\.venv\Scripts\activate

# Hoặc kích hoạt conda environment
conda activate kltn_env

# Chạy tối ưu siêu tham số
python run_optun.py --is_training 1 --root_path "../dataset" --data "custom" --data_path "mophong_data_ITS.csv" --model LSTM --target "Low,High" --task_name short_term_forecast
```

#### Chạy script tự động (Windows):
Tạo file `run_optun_windows.bat`:
```batch
@echo off
cd /d "%~dp0\MODELS"
call ..\.venv\Scripts\activate.bat
python run_optun.py --is_training 1 --root_path "../dataset" --data "custom" --data_path "mophong_data_ITS.csv" --model LSTM --target "Low,High" --task_name short_term_forecast
pause
```

Script `builtEnv.sh` (Linux) tự động thực hiện:
- Tạo môi trường ảo trong thư mục `.venv/`
- Cài đặt tất cả dependencies từ danh sách định sẵn
- Xuất danh sách packages cuối cùng ra `requirements.txt`
- Thiết lập tích hợp Jupyter kernel

## Các lệnh phát triển thường dùng

### Huấn luyện và tối ưu mô hình
```bash
# Chạy tối ưu siêu tham số Optuna
./MODELS/run_optun.sh

# Chạy với tham số cụ thể (từ thư mục MODELS)
python run_optun.py --is_training 1 --root_path "../dataset" --data "custom" --data_path "mophong_data_ITS.csv" --model LSTM --target "Low,High" --task_name short_term_forecast
```

### Các script quan trọng
- `MODELS/run_optun.py`: Script tối ưu Optuna chính kèm phân tích XAI
- `MODELS/run_args.py`: Script huấn luyện với siêu tham số thủ công
- `MODELS/run_optun.sh`: Wrapper bash cho vòng lặp huấn luyện mô hình tự động

## Tổng quan kiến trúc

### Cấu trúc thư mục
```
KLTN/
├── MODELS/                    # Huấn luyện và thí nghiệm mô hình cốt lõi
│   ├── models/               # Triển khai các mô hình (Transformer, LSTM, GRU, ...)
│   ├── layers/               # Các thành phần layer mạng neural
│   ├── exp/                  # Quản lý thí nghiệm (exp_main.py là core)
│   ├── utils/                # Hàm tiện ích và gợi ý tham số
│   ├── data/                 # Tải và xử lý dữ liệu
│   ├── checkpoints/          # Lưu trữ checkpoint mô hình
│   ├── results/              # Kết quả thí nghiệm cuối cùng
│   └── test_results/         # Output kiểm thử và trực quan hóa
├── dataset/                  # File dữ liệu CSV đầu vào
└── requirements.txt          # Dependencies Python
```

### Các thành phần cốt lõi

**Pipeline huấn luyện mô hình:**
- `exp/exp_main.py`: Class thí nghiệm chính xử lý training, validation và testing
- `utils/params.py`: Gợi ý siêu tham số cụ thể cho từng mô hình dành cho Optuna
- `utils/vis.py`: Tiện ích trực quan hóa kết quả và phân tích XAI

**Các mô hình được hỗ trợ:**
- Biến thể Transformer (Transformer, Nonstationary_Transformer, iTransformer)
- Mô hình tuần tự (LSTM, GRU)
- Mô hình tuyến tính (DLinear)

**Xử lý dữ liệu:**
- Data loader tùy chỉnh cho chuỗi thời gian giá trị khoảng
- Hỗ trợ các tác vụ dự báo với độ dài sequence và prediction có thể cấu hình
- Định dạng target: "Low,High" cho dự đoán khoảng

### Các tham số cấu hình quan trọng

**Tham số dữ liệu:**
- `--root_path`: Đường dẫn đến thư mục dataset (thường là "../dataset")
- `--data_path`: Tên file CSV trong thư mục dataset
- `--target`: Cột target (ví dụ: "Low,High" cho dự báo khoảng)
- `--task_name`: "short_term_forecast" hoặc "long_term_forecast"

**Tham số mô hình:**
- `--model`: Kiến trúc mô hình (LSTM, Transformer, DLinear, v.v.)
- `--seq_len`: Độ dài sequence đầu vào
- `--pred_len`: Độ dài sequence dự đoán
- `--features`: 'MS' (multivariate predict univariate) cho dự báo khoảng

**Tham số huấn luyện:**
- Tối ưu Optuna tự động điều chỉnh learning rate, batch size, hidden dimensions
- Kết quả lưu vào `checkpoints/` trong quá trình tối ưu, chuyển sang `results/` cho mô hình tốt nhất
- Phân tích XAI sử dụng gradient-based explanations được tạo tự động

## Quy trình phát triển

1. **Chuẩn bị dữ liệu**: Đặt file CSV vào thư mục `dataset/`
2. **Huấn luyện mô hình**: Cấu hình `MODELS/run_optun.sh` với mô hình và dataset mong muốn
3. **Tối ưu**: Chạy `./MODELS/run_optun.sh` để điều chỉnh siêu tham số tự động
4. **Kết quả**: Kiểm tra `MODELS/results/` cho mô hình cuối cùng và trực quan hóa
5. **Phân tích**: Giải thích XAI và biểu đồ hiệu suất được tạo tự động

## Thư viện phụ thuộc

Các framework chính: PyTorch, Optuna, Pandas, Matplotlib, Seaborn, Scikit-learn, SHAP (cho XAI)
Hỗ trợ GPU: NVIDIA CUDA 12.8+ với cuDNN để tăng tốc huấn luyện

## Chi tiết cấu trúc dự án

### Thư mục MODELS/ - Trung tâm huấn luyện mô hình
- **models/**: Chứa các file triển khai mô hình ML/DL
  - `Transformer.py`: Mô hình Transformer cơ bản
  - `LSTM.py`, `GRU.py`: Mô hình mạng neural tuần tự  
  - `DLinear.py`: Mô hình tuyến tính đơn giản
  - `iTransformer.py`: Inverted Transformer
  - `Nonstationary_Transformer.py`: Transformer cho dữ liệu không ổn định

- **layers/**: Các thành phần layer cấu thành mô hình
  - `Transformer_EncDec.py`: Encoder-Decoder cho Transformer
  - `SelfAttention_Family.py`: Các loại attention mechanism
  - `Embed.py`: Embedding layers
  - `utils.py`: Tiện ích layer chung

- **exp/**: Quản lý thí nghiệm và huấn luyện
  - `exp_main.py`: Class chính điều khiển toàn bộ pipeline
  - `exp_basic.py`: Chức năng thí nghiệm cơ bản
  - `exp_stats.py`: Thống kê và đánh giá

- **utils/**: Tiện ích hỗ trợ
  - `params.py`: Định nghĩa không gian siêu tham số cho Optuna
  - `vis.py`: Trực quan hóa kết quả, biểu đồ, XAI
  - `tools.py`: Công cụ chung (set seed, v.v.)

- **data/**: Xử lý và tải dữ liệu
  - Data loaders cho các định dạng dữ liệu khác nhau
  - Tiền xử lý cho chuỗi thời gian khoảng

### Quy trình hoạt động chi tiết

1. **Khởi tạo**: Script `run_optun.py` đọc tham số từ command line
2. **Tối ưu Optuna**: Tìm kiếm siêu tham số tốt nhất qua nhiều trial
3. **Huấn luyện**: Mô hình được huấn luyện với tham số tối ưu
4. **Đánh giá**: Tính toán metrics trên tập validation/test
5. **Lưu kết quả**: Checkpoint, config, và visualizations được lưu
6. **XAI Analysis**: Phân tích giải thích mô hình tự động