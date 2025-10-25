# Environment Migration Checklist - handover-rl

**目的**: 確保在新環境中可以完整復現 handover-rl 專案

**最後更新**: 2025-10-25

---

## ✅ Git 追蹤狀況檢查

### 1. Git Status
```bash
cd /home/sat/satellite/handover-rl
git status
```

**檢查結果** (2025-10-25):
- ✅ **151 個檔案已追蹤**
- ✅ **工作目錄乾淨** (無未提交變更)
- ✅ **無未追蹤的重要檔案**

### 2. 已追蹤的重要檔案
- ✅ **源代碼**: `src/` 所有 Python 檔案
- ✅ **配置檔案**: `config/*.yaml`, `.env.example`
- ✅ **文檔**: `docs/` 所有文檔
- ✅ **腳本**: `*.sh` 所有執行腳本
- ✅ **依賴清單**: `requirements.txt`
- ✅ **專案檔案**: `README.md`, `LICENSE`, `setup.py`

### 3. 正確忽略的檔案 (.gitignore)
- ✅ **虛擬環境**: `venv/`, `env/`, `.venv/`
- ✅ **訓練輸出**:
  - `checkpoints/*.pth` (保留 .gitkeep)
  - `output/` (保留 .gitkeep)
  - `results/` (保留 .gitkeep)
  - `logs/` (保留 .gitkeep)
- ✅ **生成數據**:
  - `data/episodes/`
  - `data/*.json`, `data/*.npz`
  - `data/*.hdf5` (保留 .gitkeep)
- ✅ **Python 緩存**: `__pycache__/`, `*.pyc`, `.pytest_cache/`
- ✅ **編輯器**: `.vscode/`, `.idea/`
- ✅ **環境變數**: `.env` (僅追蹤 `.env.example`)

---

## ✅ Requirements.txt 完整性檢查

### 1. requirements.txt 內容 (28 個核心套件)

**科學計算** (來自 orbit-engine):
```
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
astropy>=5.2.0
skyfield>=1.45
sgp4>=2.22
h5py>=3.8.0
pyproj>=3.5.0
```

**深度學習 & RL**:
```
torch>=2.0.0
torchaudio>=2.0.0
torchvision>=0.15.0
gymnasium>=1.0.0
```

**訊號處理**:
```
itur>=0.4.0
```

**可視化 & 日誌**:
```
matplotlib>=3.7.0
pillow>=10.0.0
tensorboard>=2.13.0
tqdm>=4.65.0
```

**工具套件**:
```
python-dotenv>=1.0.0
pydantic>=2.0.0
PyYAML>=6.0
requests>=2.31.0
httpx>=0.24.0
```

**測試**:
```
pytest>=7.3.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0
```

**時間處理**:
```
python-dateutil>=2.8.0
pytz>=2023.3
```

**系統監控**:
```
psutil>=5.9.0
numpy-financial>=1.0.0
```

### 2. Virtual Environment 套件對比

**venv 安裝套件總數**: 90 個
- **核心依賴** (requirements.txt): 28 個
- **傳遞性依賴**: 62 個

**傳遞性依賴分類**:
- **NVIDIA CUDA** (15 個): nvidia-cublas-cu12, nvidia-cudnn-cu12, etc. (來自 torch)
- **PyTorch 依賴** (6 個): triton, sympy, networkx, mpmath, filelock, fsspec
- **Matplotlib 依賴** (4 個): contourpy, cycler, fonttools, kiwisolver
- **TensorBoard 依賴** (9 個): grpcio, Markdown, Jinja2, Werkzeug, protobuf, etc.
- **HTTP/Async** (4 個): httpcore, h11, anyio, sniffio (來自 httpx)
- **Pydantic 依賴** (4 個): annotated-types, pydantic_core, typing_extensions
- **科學計算** (3 個): pyerfa (from astropy), jplephem (from skyfield), pyproj
- **測試依賴** (3 個): pluggy, iniconfig, coverage
- **通用工具** (14 個): certifi, charset-normalizer, idna, urllib3, packaging, six, etc.

**✅ 結論**: 所有 62 個額外套件都是合法的傳遞性依賴，無需清理

### 3. 套件版本驗證 (當前 venv)

**關鍵套件版本** (2025-10-25):
```
torch==2.9.0
gymnasium==1.2.1
numpy==2.3.4
pandas==2.3.3
astropy==7.1.1
skyfield==1.53
h5py==3.15.1
matplotlib==3.10.7
tensorboard==2.20.0
pydantic==2.12.3
pytest==8.4.2
```

**✅ 所有套件版本符合 requirements.txt 最低版本要求**

---

## 🚀 新環境部署步驟

### 1. Virtual Environment 部署

**克隆專案**:
```bash
git clone https://github.com/yourusername/handover-rl.git
cd handover-rl
```

**檢查 orbit-engine 依賴**:
```bash
# 確認 orbit-engine 已安裝在 ../orbit-engine
ls ../orbit-engine
```

**設置虛擬環境**:
```bash
# 使用專案提供的設置腳本
./setup_env.sh all

# 或手動設置
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**驗證安裝**:
```bash
# 檢查關鍵套件
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"
python -c "import h5py; print(f'h5py: {h5py.__version__}')"
```

### 2. 環境變數配置

**複製 .env.example**:
```bash
cp .env.example .env
```

**編輯 .env** (必要配置):
```bash
# orbit-engine 路徑
ORBIT_ENGINE_PATH=../orbit-engine

# 訓練配置
HANDOVER_RL_DEVICE=cuda  # 或 cpu
HANDOVER_RL_NUM_WORKERS=4

# 數據路徑 (可選，使用預設值即可)
# HANDOVER_RL_DATA_PATH=data/
# HANDOVER_RL_OUTPUT_PATH=output/
```

### 3. 目錄結構初始化

**創建必要目錄**:
```bash
# 腳本會自動創建這些目錄
./setup_env.sh all

# 或手動創建
mkdir -p data/episodes
mkdir -p checkpoints
mkdir -p output
mkdir -p results
mkdir -p logs
```

**驗證目錄結構**:
```bash
tree -L 2 -I 'venv|__pycache__|*.pyc'
```

預期結構:
```
handover-rl/
├── checkpoints/          # 訓練模型檢查點
│   └── .gitkeep
├── config/               # 配置檔案
│   ├── data_gen_config.yaml
│   └── training_config.yaml
├── data/                 # 訓練數據
│   ├── episodes/        # (生成時創建)
│   └── .gitkeep
├── docs/                 # 文檔
├── logs/                 # 訓練日誌
│   └── .gitkeep
├── output/               # 訓練輸出
│   └── .gitkeep
├── results/              # 評估結果
│   └── .gitkeep
├── src/                  # 源代碼
│   ├── agents/
│   ├── environments/
│   ├── strategies/
│   └── utils/
├── .env.example          # 環境變數範本
├── requirements.txt      # Python 依賴
├── setup_env.sh         # 環境設置腳本
└── README.md            # 專案說明
```

### 4. 從 orbit-engine 獲取訓練數據

**確認 orbit-engine Stage 4 輸出**:
```bash
# 檢查 Elite Pool 數據是否存在
ls -lh ../orbit-engine/data/outputs/stage4_*.json
```

**預期檔案**:
- `stage4_analysis_*.json` - Stage 4 分析結果
- `stage4_connectable_satellites_*.json` - Elite Pool (129 顆衛星)

**生成 HDF5 訓練數據** (如果需要):
```bash
# 在 orbit-engine 中執行 Stage 5-6
cd ../orbit-engine
./run.sh stage5 stage6

# 檢查生成的 HDF5 檔案
ls -lh data/outputs/rl_training/stage6/*.hdf5
```

### 5. 快速驗證測試

**Level 0 煙霧測試** (10 分鐘):
```bash
source venv/bin/activate
./quick_train.sh 0
```

**預期輸出**:
- ✅ 環境初始化成功
- ✅ 加載 Elite Pool 數據 (129 顆衛星)
- ✅ DQN 模型創建成功
- ✅ 訓練運行 10 集 (episodes)
- ✅ 模型檢查點保存成功

**驗證生成的檔案**:
```bash
# 檢查訓練日誌
ls -lh logs/

# 檢查模型檢查點
ls -lh checkpoints/

# 檢查 TensorBoard 日誌
ls -lh output/
```

---

## 🐳 Docker 部署驗證

### 1. Dockerfile 檢查

**確認 Dockerfile 存在**:
```bash
ls -l Dockerfile docker-compose.yml
```

### 2. Docker 構建

**構建映像**:
```bash
docker build -t handover-rl:latest .
```

**驗證映像**:
```bash
docker images | grep handover-rl
```

### 3. Docker 運行測試

**運行 Level 0 測試**:
```bash
docker run --rm \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/logs:/app/logs \
  handover-rl:latest \
  ./quick_train.sh 0
```

**使用 GPU** (如果有 NVIDIA Docker):
```bash
docker run --rm --gpus all \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/logs:/app/logs \
  handover-rl:latest \
  ./quick_train.sh 0
```

---

## 🔍 常見問題排查

### 1. Import Error: No module named 'gymnasium'

**原因**: 虛擬環境未激活或 requirements.txt 未安裝

**解決方法**:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 2. FileNotFoundError: orbit-engine data not found

**原因**: orbit-engine 未在正確路徑或未生成 Stage 4 數據

**解決方法**:
```bash
# 檢查 orbit-engine 位置
ls ../orbit-engine

# 生成 Stage 4 數據
cd ../orbit-engine
./run.sh stage1 stage2 stage3 stage4
```

### 3. CUDA out of memory

**原因**: GPU 記憶體不足

**解決方法**:
```bash
# 方案 1: 使用 CPU
export HANDOVER_RL_DEVICE=cpu

# 方案 2: 減少批次大小
# 編輯 config/training_config.yaml
# batch_size: 64 → 32
```

### 4. 訓練速度過慢

**原因**: 未使用 GPU 或數據加載效率問題

**解決方法**:
```bash
# 檢查 PyTorch CUDA 可用性
python -c "import torch; print(torch.cuda.is_available())"

# 調整 worker 數量
# 編輯 .env
# HANDOVER_RL_NUM_WORKERS=4
```

---

## 📋 部署檢查清單

**在新環境部署時，請依序檢查以下項目**:

### Git & 代碼
- [ ] Git clone 成功
- [ ] 所有源代碼檔案完整
- [ ] .env.example 存在
- [ ] setup_env.sh 存在且可執行

### Python 環境
- [ ] Python 3.10+ 已安裝
- [ ] venv 創建成功
- [ ] requirements.txt 安裝成功 (28 個核心套件)
- [ ] PyTorch 正確安裝並可用
- [ ] Gymnasium 版本 >= 1.0.0

### 目錄結構
- [ ] data/ 目錄存在
- [ ] checkpoints/ 目錄存在
- [ ] output/ 目錄存在
- [ ] logs/ 目錄存在
- [ ] 所有 .gitkeep 檔案存在

### 依賴檢查
- [ ] orbit-engine 在 ../orbit-engine
- [ ] orbit-engine Stage 4 數據已生成
- [ ] .env 檔案已配置

### 功能驗證
- [ ] Level 0 測試運行成功 (10 min)
- [ ] TensorBoard 日誌正常
- [ ] 模型檢查點正常保存
- [ ] (可選) Docker 構建成功
- [ ] (可選) Docker 運行測試成功

---

## 📊 磁碟空間需求

**最小需求**: 2GB
**建議需求**: 10GB (包含訓練數據和模型)

**目錄大小參考** (2025-10-25):
```
venv/            7.3 GB  (虛擬環境)
checkpoints/     6.5 MB  (訓練模型)
output/          6.0 MB  (TensorBoard 日誌)
results/         568 KB  (評估結果)
data/            變動    (依訓練等級而定)
logs/            變動    (依訓練時長而定)
```

---

## 🔗 相關文檔

- **[README.md](README.md)** - 專案總覽
- **[Quick Start Guide](docs/training/QUICKSTART.md)** - 快速開始指南
- **[Training Levels](docs/training/TRAINING_LEVELS.md)** - 訓練等級說明
- **[Data Dependencies](docs/architecture/DATA_DEPENDENCIES.md)** - 數據依賴說明

---

## ✅ 驗證完成

**檢查日期**: 2025-10-25

**檢查項目**:
- ✅ Git 追蹤狀況正常 (151 個檔案)
- ✅ .gitignore 配置完整
- ✅ requirements.txt 包含所有必要套件 (28 個)
- ✅ venv 傳遞性依賴正確 (62 個額外套件皆合法)
- ✅ 無多餘或缺失套件

**結論**: handover-rl 專案已準備好在新環境中部署

**部署建議**:
1. 使用 `./setup_env.sh all` 自動化設置
2. 確保 orbit-engine 已正確安裝在 ../orbit-engine
3. 先運行 Level 0 測試驗證環境正確性
4. 根據需求選擇合適的訓練等級 (1, 3, 5)

---

**維護者**: SuperClaude
**最後驗證**: 2025-10-25
**專案版本**: 1.0.0-dev
