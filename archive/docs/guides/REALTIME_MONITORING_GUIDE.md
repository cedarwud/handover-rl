## 實時動態監控系統 - 完整指南

您完全正確！靜態圖表（PDF/PNG）只適合論文最終發表，**訓練過程中應該要有實時動態監控**！

我們提供了 **3 種實時監控方案**，讓您能夠在訓練過程中即時看到動態變化的圖表。

---

## 📊 三種方案對比

| 方案 | 技術 | 優點 | 缺點 | 推薦場景 |
|-----|------|------|------|---------|
| **TensorBoard** ⭐ | TensorBoard | RL 社群標準<br>功能完整<br>易於使用 | 需要額外終端 | **訓練監控**<br>（最推薦） |
| **Web Dashboard** | Flask + Plotly | 美觀現代<br>高度互動<br>自定義靈活 | 需要 Python 服務 | 演示<br>展示 |
| **自動刷新 HTML** | 靜態 HTML | 無需服務器<br>最輕量<br>易於分享 | 功能較簡單 | 快速查看<br>遠程監控 |

---

## 🚀 快速開始

### 方法 1: 一鍵啟動（推薦）

```bash
# TensorBoard 監控（推薦）
./start_monitor.sh tensorboard

# Web Dashboard
./start_monitor.sh dashboard

# 自動刷新 HTML
./start_monitor.sh html
```

### 方法 2: 手動啟動

見下方各方案的詳細說明。

---

## 📈 方案 1: TensorBoard 監控 ⭐ 推薦

### 為什麼選擇 TensorBoard？

- ✅ **RL 社群標準** - 幾乎所有 RL 論文都使用
- ✅ **功能完整** - 支持標量、直方圖、圖像等
- ✅ **易於使用** - 簡單命令即可啟動
- ✅ **性能優秀** - 處理大量數據無壓力

### 啟動方法

#### 步驟 1: 啟動數據轉換器

```bash
# 後台運行數據轉換器
python scripts/realtime_tensorboard.py training_level5_20min_final.log &
```

這會持續監控訓練日誌，並將數據寫入 TensorBoard 格式。

#### 步驟 2: 啟動 TensorBoard

```bash
tensorboard --logdir=logs/tensorboard --port=6006
```

#### 步驟 3: 瀏覽器訪問

打開瀏覽器訪問：
```
http://localhost:6006
```

### 功能展示

TensorBoard 提供以下實時圖表：

1. **Training/Reward_Mean** - Episode Reward 平均值
2. **Training/Reward_Std** - Episode Reward 標準差
3. **Training/Loss** - 訓練 Loss
4. **Training/Handovers_Mean** - Handover 頻率平均值
5. **Training/Progress** - 訓練進度百分比

### 特點

- 🔄 **實時更新** - 每 10 秒自動刷新
- 📊 **平滑選項** - 可調整曲線平滑度
- 📥 **數據下載** - 可下載 CSV 格式數據
- 🔍 **縮放功能** - 可縮放查看細節
- 📌 **標記功能** - 可標記重要時刻

### 截圖示意

```
┌─────────────────────────────────────────────────────┐
│  TensorBoard - Training Metrics                     │
├─────────────────────────────────────────────────────┤
│                                                       │
│  Training/Reward_Mean        Training/Loss          │
│  ┌─────────────────┐          ┌─────────────────┐  │
│  │      ／          │          │  ＼              │  │
│  │    ／            │          │    ＼            │  │
│  │  ／              │          │      ＼__        │  │
│  └─────────────────┘          └─────────────────┘  │
│                                                       │
│  Training/Handovers_Mean    Training/Progress       │
│  ┌─────────────────┐          ┌─────────────────┐  │
│  │  ～～～～～       │          │    ／／／        │  │
│  │  ～～～～～       │          │  ／／／          │  │
│  │  ～～～～～       │          │／／／            │  │
│  └─────────────────┘          └─────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## 🌐 方案 2: Web Dashboard

### 為什麼選擇 Web Dashboard？

- ✅ **現代美觀** - 精心設計的 UI
- ✅ **高度互動** - 基於 Plotly，支持縮放、懸停等
- ✅ **自動刷新** - 每 5 秒自動更新
- ✅ **響應式設計** - 支持手機、平板訪問

### 啟動方法

```bash
python scripts/realtime_dashboard.py training_level5_20min_final.log
```

或使用快捷腳本：

```bash
./start_monitor.sh dashboard
```

### 訪問地址

```
http://localhost:5000
```

如果在遠程服務器上運行，可訪問：

```
http://your-server-ip:5000
```

### 界面預覽

```
┌─────────────────────────────────────────────────────────┐
│  🛰️ LEO Satellite Handover Training Monitor            │
│  Real-time training metrics dashboard                   │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │ RUNNING │ │ Ep: 23  │ │ 1.3%    │ │ R:-648  │       │
│  │ Status  │ │ /1700   │ │ Progress│ │ Reward  │       │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │
│                                                           │
│  📈 Episode Reward          📉 Training Loss             │
│  ┌────────────────────┐    ┌────────────────────┐       │
│  │   動態互動圖表     │    │   動態互動圖表     │       │
│  │   (Plotly)         │    │   (Plotly)         │       │
│  └────────────────────┘    └────────────────────┘       │
│                                                           │
│  🔄 Handover Frequency      📊 Training Progress         │
│  ┌────────────────────┐    ┌────────────────────┐       │
│  │   動態互動圖表     │    │   動態互動圖表     │       │
│  └────────────────────┘    └────────────────────┘       │
│                                                           │
│  Last updated: 06:30:15 | Auto-refresh every 5 seconds  │
└─────────────────────────────────────────────────────────┘
```

### 特點

- 🎨 **深色主題** - 保護眼睛，專業外觀
- 📊 **6 個統計卡片** - 關鍵指標一目了然
- 📈 **4 個動態圖表** - 完整覆蓋訓練指標
- 🔄 **自動刷新** - 每 5 秒更新
- 📱 **響應式** - 支持各種螢幕尺寸

---

## 📄 方案 3: 自動刷新 HTML

### 為什麼選擇自動刷新 HTML？

- ✅ **最輕量** - 無需額外服務
- ✅ **易於分享** - 直接發送 HTML 文件
- ✅ **遠程友好** - 適合通過 SSH 的場景
- ✅ **簡單可靠** - 最少依賴

### 啟動方法

```bash
# 後台持續生成
python scripts/generate_live_html.py training_level5_20min_final.log &

# 用瀏覽器打開
firefox live_monitor.html
# 或
chromium live_monitor.html
```

或使用快捷腳本：

```bash
./start_monitor.sh html
```

### 工作原理

1. 腳本每 10 秒讀取訓練日誌
2. 生成/更新 `live_monitor.html`
3. HTML 內有 `<meta http-equiv="refresh" content="10">`
4. 瀏覽器每 10 秒自動刷新頁面

### 特點

- 📄 **靜態 HTML** - 可隨意複製、分享
- 🔄 **自動刷新** - 瀏覽器每 10 秒刷新
- 📊 **Plotly 圖表** - 互動式圖表
- 💾 **低資源消耗** - 幾乎無性能影響

### 遠程訪問

如果在遠程服務器上運行，可以：

```bash
# 在遠程服務器上生成 HTML
python scripts/generate_live_html.py training.log

# 用 scp 下載到本地
scp user@server:/path/to/live_monitor.html .

# 本地瀏覽器打開
firefox live_monitor.html
```

---

## 🔧 高級用法

### 自定義端口

#### TensorBoard

```bash
tensorboard --logdir=logs/tensorboard --port=8888
```

#### Web Dashboard

```bash
python scripts/realtime_dashboard.py training.log --port=8080
```

### 自定義更新頻率

#### TensorBoard 數據轉換器

```bash
python scripts/realtime_tensorboard.py training.log --interval 5  # 5 秒更新
```

#### HTML 報告

```bash
python scripts/generate_live_html.py training.log --interval 30  # 30 秒更新
```

### 遠程訪問設置

#### Web Dashboard 允許外部訪問

```bash
python scripts/realtime_dashboard.py training.log --host 0.0.0.0 --port 5000
```

然後從其他電腦訪問：
```
http://your-server-ip:5000
```

⚠️ **安全提示**: 如果開放外部訪問，建議配置防火牆規則或使用 SSH 隧道。

### SSH 隧道（推薦）

如果訓練在遠程服務器上，使用 SSH 隧道最安全：

```bash
# 本地執行
ssh -L 6006:localhost:6006 user@server

# 然後在本地瀏覽器訪問
http://localhost:6006
```

---

## 📊 與靜態圖表的關係

### 實時監控 vs 靜態圖表

| 用途 | 工具 | 時機 |
|-----|------|------|
| **訓練監控** | TensorBoard<br>Web Dashboard<br>自動刷新 HTML | 訓練進行中 |
| **論文發表** | `generate_paper_figures.sh` | 訓練完成後 |

### 工作流程

```
1. 開始訓練
   ↓
2. 啟動實時監控 (TensorBoard / Dashboard / HTML)
   ├─ 持續觀察訓練進度
   ├─ 及時發現問題
   └─ 驗證數值穩定性
   ↓
3. Episode 920 到達
   ├─ 在實時監控中觀察 loss
   └─ 確認 loss < 10 (驗證 bug 修復)
   ↓
4. 訓練完成
   ↓
5. 生成論文圖表
   └─ ./generate_paper_figures.sh
   ↓
6. 插入論文中
```

---

## 💡 最佳實踐

### 推薦組合

#### 本地訓練

```bash
# 終端 1: 訓練
./train_level5_final.sh

# 終端 2: TensorBoard
./start_monitor.sh tensorboard
```

#### 遠程服務器訓練

**選項 A: SSH 隧道 + TensorBoard**

```bash
# 遠程服務器
./train_level5_final.sh
python scripts/realtime_tensorboard.py training.log &
tensorboard --logdir=logs/tensorboard --port=6006

# 本地電腦
ssh -L 6006:localhost:6006 user@server

# 本地瀏覽器
http://localhost:6006
```

**選項 B: 自動刷新 HTML（最簡單）**

```bash
# 遠程服務器
./train_level5_final.sh
python scripts/generate_live_html.py training.log &

# 定期下載 HTML 到本地
scp user@server:~/handover-rl/live_monitor.html .
# 本地瀏覽器打開
```

### 監控檢查清單

訓練期間應該監控的指標：

- [ ] **Reward** 是否逐漸提升？
- [ ] **Loss** 是否保持穩定（< 100）？
- [ ] **Episode 920** loss 是否 < 10？（關鍵檢查）
- [ ] **Handovers** 頻率是否合理（10-30 次）？
- [ ] **Progress** 是否線性增長？
- [ ] 是否有 NaN/Inf 出現？

---

## 🐛 故障排除

### TensorBoard 問題

**問題**: TensorBoard 顯示 "No dashboards are active"

**解決**:
```bash
# 檢查數據轉換器是否運行
ps aux | grep realtime_tensorboard

# 檢查 TensorBoard 目錄
ls -lh logs/tensorboard/

# 重新啟動
pkill -f realtime_tensorboard
python scripts/realtime_tensorboard.py training.log &
tensorboard --logdir=logs/tensorboard
```

### Web Dashboard 問題

**問題**: 無法訪問 Dashboard

**解決**:
```bash
# 檢查 Flask 是否運行
ps aux | grep realtime_dashboard

# 檢查端口是否被占用
netstat -tulpn | grep 5000

# 使用不同端口
python scripts/realtime_dashboard.py training.log --port 5001
```

### HTML 問題

**問題**: HTML 不更新

**解決**:
```bash
# 檢查生成器是否運行
ps aux | grep generate_live_html

# 檢查 HTML 文件時間戳
ls -lh live_monitor.html

# 重新啟動
pkill -f generate_live_html
python scripts/generate_live_html.py training.log &
```

### 通用問題

**問題**: 圖表無數據

**原因**: 訓練日誌可能還沒有數據點

**解決**: 等待訓練到第 10 個 episode（第一個報告點）

---

## 📚 參考資源

### TensorBoard 文檔

- [官方文檔](https://www.tensorflow.org/tensorboard)
- [PyTorch TensorBoard 教程](https://pytorch.org/docs/stable/tensorboard.html)

### Plotly 文檔

- [Plotly Python](https://plotly.com/python/)
- [Plotly.js](https://plotly.com/javascript/)

### Flask 文檔

- [官方文檔](https://flask.palletsprojects.com/)

---

## ✅ 總結

您現在有 **3 種實時監控方案**：

### 🥇 TensorBoard（推薦）
- RL 社群標準
- 功能最完整
- 最適合訓練監控

### 🥈 Web Dashboard
- 最美觀
- 高度互動
- 最適合演示展示

### 🥉 自動刷新 HTML
- 最輕量
- 無需服務
- 最適合遠程監控

**快速啟動**：

```bash
# 推薦：TensorBoard
./start_monitor.sh tensorboard

# 或：Web Dashboard
./start_monitor.sh dashboard

# 或：自動刷新 HTML
./start_monitor.sh html
```

**論文圖表**（訓練完成後）：

```bash
./generate_paper_figures.sh
```

---

## 🎯 現在就試試！

```bash
# 立即啟動實時監控
./start_monitor.sh tensorboard
```

享受實時動態的訓練監控體驗！🚀📊
