# 訓練監控系統整合指南

將實時訓練監控整合到您現有的 **leo-simulator (前端)** 和 **orbit-engine (後端)** 架構中。

---

## 📋 目錄

1. [架構概覽](#架構概覽)
2. [後端整合 (orbit-engine)](#後端整合-orbit-engine)
3. [前端整合 (leo-simulator)](#前端整合-leo-simulator)
4. [部署配置](#部署配置)
5. [測試驗證](#測試驗證)

---

## 🏗️ 架構概覽

### 現有架構

```
leo-simulator (前端)
├── React 18 + TypeScript
├── React Three Fiber (3D)
├── Vite 7.1.12
└── 端口: 5173 (開發)

orbit-engine (後端)
├── Python 3.13
├── Skyfield (SGP4)
├── FastAPI (推測)
└── 端口: 8000 (推測)
```

### 整合後架構

```
┌─────────────────────────────────────────┐
│  leo-simulator (前端)                    │
│  ├── 原有 3D 可視化                      │
│  └── 新增: TrainingMonitor 組件         │
│       ↓ HTTP API                        │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│  orbit-engine (後端)                     │
│  ├── 原有 SGP4 軌道計算                  │
│  └── 新增: /api/training/* 路由         │
│       ↓ 讀取                             │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│  handover-rl (訓練)                      │
│  └── training_level5_20min_final.log    │
└─────────────────────────────────────────┘
```

---

## 🐍 後端整合 (orbit-engine)

### 方案 A: 整合到現有 FastAPI App（推薦）

如果 orbit-engine 已經使用 FastAPI：

#### 步驟 1: 複製 API 模塊

```bash
# 從 handover-rl 複製 API 模塊到 orbit-engine
cp -r handover-rl/api orbit-engine/src/
```

#### 步驟 2: 在 orbit-engine 主應用中整合

**orbit-engine/src/main.py** (或您的主 FastAPI 文件):

```python
from fastapi import FastAPI
from api.training_monitor_api import router as training_router

app = FastAPI(title="Orbit Engine API")

# 原有路由
@app.get("/")
async def root():
    return {"message": "Orbit Engine API"}

# ... 其他原有路由 ...

# 🆕 新增: 訓練監控路由
app.include_router(
    training_router,
    prefix="/api/training",
    tags=["training"]
)
```

#### 步驟 3: 配置日誌路徑

在 `api/training_monitor_api.py` 的 `startup_event` 中：

```python
@app.on_event("startup")
async def startup_event():
    global monitor
    # 🔧 配置實際的訓練日誌路徑
    log_file = "../handover-rl/training_level5_20min_final.log"
    # 或使用環境變數
    # log_file = os.getenv("TRAINING_LOG_PATH", "training.log")

    if Path(log_file).exists():
        monitor = TrainingDataMonitor(log_file)
        print(f"✅ Training monitor initialized: {log_file}")
```

#### 步驟 4: 安裝依賴

```bash
cd orbit-engine
pip install fastapi uvicorn pydantic
```

#### 步驟 5: 啟動服務

```bash
# orbit-engine 目錄
uvicorn src.main:app --reload --port 8000
```

現在 API 可在以下訪問：
- http://localhost:8000/api/training/status
- http://localhost:8000/api/training/metrics
- http://localhost:8000/docs (API 文檔)

---

### 方案 B: 獨立運行（簡單快速）

如果不想修改 orbit-engine，可以獨立運行：

```bash
cd handover-rl
uvicorn api.training_monitor_api:app --port 8001
```

然後在前端配置 `apiBaseUrl="http://localhost:8001"`

---

## ⚛️ 前端整合 (leo-simulator)

### 步驟 1: 複製組件文件

```bash
# 從 handover-rl 複製前端組件到 leo-simulator
cp handover-rl/frontend/TrainingMonitor.tsx leo-simulator/src/components/
cp handover-rl/frontend/TrainingMonitor.css leo-simulator/src/components/
```

### 步驟 2: 安裝依賴

**leo-simulator/package.json**:

```bash
cd leo-simulator
npm install recharts
# 或
pnpm add recharts
```

`recharts` 用於圖表渲染（React 圖表庫）。

### 步驟 3: 在應用中使用

**leo-simulator/src/App.tsx** (或任何您想放置的地方):

```typescript
import { TrainingMonitor } from './components/TrainingMonitor';
import './components/TrainingMonitor.css';

function App() {
  return (
    <div className="app">
      {/* 原有的 3D 可視化等組件 */}
      <YourExisting3DView />

      {/* 🆕 新增: 訓練監控面板 */}
      <TrainingMonitor
        apiBaseUrl="http://localhost:8000/api/training"  // 整合版
        // 或
        // apiBaseUrl="http://localhost:8001"  // 獨立運行版
        refreshInterval={5000}  // 5 秒刷新
        showCharts={true}
      />
    </div>
  );
}
```

### 步驟 4: 樣式定制（可選）

根據 leo-simulator 的設計系統調整 `TrainingMonitor.css`：

```css
/* 使用 leo-simulator 的配色 */
.training-monitor {
  background: var(--your-bg-color);
  color: var(--your-text-color);
}
```

### 步驟 5: 添加 Tab 或 Modal（推薦）

如果不想總是顯示，可以做成 Tab 或 Modal：

```typescript
import { useState } from 'react';
import { TrainingMonitor } from './components/TrainingMonitor';

function App() {
  const [showMonitor, setShowMonitor] = useState(false);

  return (
    <div className="app">
      {/* 切換按鈕 */}
      <button onClick={() => setShowMonitor(!showMonitor)}>
        {showMonitor ? 'Hide' : 'Show'} Training Monitor
      </button>

      {/* 條件渲染監控面板 */}
      {showMonitor && (
        <div className="monitor-panel">
          <TrainingMonitor apiBaseUrl="http://localhost:8000/api/training" />
        </div>
      )}

      {/* 原有組件 */}
      <YourExisting3DView />
    </div>
  );
}
```

---

## 🔧 部署配置

### 開發環境

**CORS 設置** (已在 API 中配置):

`api/training_monitor_api.py` 已設置允許所有來源：

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開發環境
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 生產環境

#### 後端 (orbit-engine)

**限制 CORS 來源**:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-leo-simulator.com",
        "https://your-domain.com"
    ],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)
```

**使用環境變數配置**:

```python
import os

@app.on_event("startup")
async def startup_event():
    log_file = os.getenv("TRAINING_LOG_PATH", "training.log")
    monitor = TrainingDataMonitor(log_file)
```

```bash
# 啟動時設置
export TRAINING_LOG_PATH=/path/to/training.log
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

#### 前端 (leo-simulator)

**使用環境變數**:

**leo-simulator/.env**:

```bash
VITE_TRAINING_API_URL=https://api.your-domain.com/api/training
```

**在代碼中使用**:

```typescript
<TrainingMonitor
  apiBaseUrl={import.meta.env.VITE_TRAINING_API_URL}
/>
```

---

## 🧪 測試驗證

### 測試後端 API

#### 1. 測試獨立運行

```bash
cd handover-rl
uvicorn api.training_monitor_api:app --reload --port 8001
```

訪問 http://localhost:8001/docs 查看 API 文檔。

#### 2. 測試端點

```bash
# 獲取狀態
curl http://localhost:8001/status

# 獲取所有指標
curl http://localhost:8001/metrics

# 獲取最新數據
curl http://localhost:8001/latest
```

預期響應：

```json
{
  "episode": 23,
  "total_episodes": 1700,
  "reward_mean": -648.07,
  "reward_std": 732.17,
  "handovers_mean": 26.4,
  "handovers_std": 25.0,
  "loss": 5.2182,
  "timestamp": "2025-11-03T06:30:00"
}
```

### 測試前端組件

#### 1. 在 leo-simulator 中測試

```bash
cd leo-simulator
npm run dev
# 或
pnpm dev
```

訪問 http://localhost:5173

#### 2. 檢查瀏覽器控制台

- 無 CORS 錯誤
- 能看到 API 請求成功
- 數據正確顯示

#### 3. 測試功能

- [ ] 狀態卡片顯示正確
- [ ] 圖表能正確渲染
- [ ] 每 5 秒自動刷新
- [ ] 數據更新時圖表動態變化
- [ ] 錯誤處理正常（斷開後端測試）

---

## 🎨 UI/UX 建議

### 與 3D 可視化整合

#### 方案 A: 側邊欄

```typescript
<div className="app-layout">
  <aside className="sidebar">
    <TrainingMonitor apiBaseUrl="..." />
  </aside>

  <main className="main-view">
    <Your3DVisualization />
  </main>
</div>
```

#### 方案 B: 可折疊面板

```typescript
const [expanded, setExpanded] = useState(false);

<div className={`monitor-panel ${expanded ? 'expanded' : 'collapsed'}`}>
  <button onClick={() => setExpanded(!expanded)}>
    {expanded ? '▼' : '▶'} Training Monitor
  </button>

  {expanded && <TrainingMonitor apiBaseUrl="..." />}
</div>
```

#### 方案 C: Modal 彈窗

```typescript
import { Modal } from 'your-ui-library';

<Modal open={showMonitor} onClose={() => setShowMonitor(false)}>
  <TrainingMonitor apiBaseUrl="..." />
</Modal>
```

### 響應式設計

組件已內建響應式設計，在手機/平板上會自動調整佈局。

---

## 📦 完整範例

### 最小整合範例

**orbit-engine/src/main.py**:

```python
from fastapi import FastAPI
from api.training_monitor_api import router as training_router

app = FastAPI()

app.include_router(training_router, prefix="/api/training")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**leo-simulator/src/App.tsx**:

```typescript
import { TrainingMonitor } from './components/TrainingMonitor';
import './components/TrainingMonitor.css';

function App() {
  return (
    <div className="app">
      <h1>LEO Simulator</h1>

      <TrainingMonitor
        apiBaseUrl="http://localhost:8000/api/training"
        refreshInterval={5000}
      />
    </div>
  );
}

export default App;
```

**啟動**:

```bash
# 終端 1: 後端
cd orbit-engine
uvicorn src.main:app --reload --port 8000

# 終端 2: 前端
cd leo-simulator
pnpm dev

# 終端 3: 訓練（如果還沒開始）
cd handover-rl
./train_level5_final.sh
```

---

## 🔍 故障排除

### 問題 1: CORS 錯誤

**錯誤訊息**:
```
Access to fetch at 'http://localhost:8000/api/training/metrics' from origin 'http://localhost:5173' has been blocked by CORS policy
```

**解決**:

確認後端 CORS 設置：

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 問題 2: API 無響應

**檢查**:

1. 後端是否運行？
   ```bash
   curl http://localhost:8000/api/training/status
   ```

2. 訓練日誌是否存在？
   ```bash
   ls -lh training_level5_20min_final.log
   ```

3. 日誌路徑配置是否正確？

### 問題 3: 圖表不顯示

**檢查**:

1. `recharts` 是否已安裝？
   ```bash
   npm list recharts
   ```

2. 是否有數據？檢查 API 響應

3. 瀏覽器控制台是否有錯誤？

---

## 📚 API 文檔

### 端點列表

| 端點 | 方法 | 說明 |
|------|------|------|
| `/api/training/status` | GET | 訓練狀態 |
| `/api/training/episodes` | GET | 所有 Episode 數據 |
| `/api/training/latest` | GET | 最新 Episode 數據 |
| `/api/training/metrics` | GET | 完整訓練指標 |

### 響應格式

詳見：http://localhost:8000/docs (FastAPI 自動生成的文檔)

---

## ✅ 檢查清單

### 後端整合

- [ ] 複製 `api/training_monitor_api.py` 到 orbit-engine
- [ ] 在主 FastAPI app 中添加 router
- [ ] 配置訓練日誌路徑
- [ ] 安裝依賴 (fastapi, uvicorn)
- [ ] 測試 API 端點
- [ ] 配置 CORS

### 前端整合

- [ ] 複製 `TrainingMonitor.tsx` 和 `.css` 到 leo-simulator
- [ ] 安裝 `recharts`
- [ ] 在 App 中導入組件
- [ ] 配置 `apiBaseUrl`
- [ ] 測試顯示和刷新
- [ ] 根據設計系統調整樣式

### 生產部署

- [ ] 限制 CORS 來源
- [ ] 使用環境變數配置
- [ ] 設置 HTTPS
- [ ] 監控 API 性能
- [ ] 日誌輪替設置

---

## 🎯 下一步

整合完成後，您將擁有：

✅ **實時訓練監控** - 在 leo-simulator 中直接查看訓練進度
✅ **統一架構** - 與現有 orbit-engine 無縫整合
✅ **3D + 訓練** - 同時監控軌道和訓練
✅ **生產就緒** - FastAPI + React 現代技術棧

祝整合順利！🚀
