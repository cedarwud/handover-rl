# Precompute Architecture Decision

**Date**: 2025-11-08
**Decision**: 預計算系統的位置選擇

---

## 問題

預計算系統 (OrbitPrecomputeGenerator/Table) 應該放在哪裡？

1. **保留在 handover-rl/** (當前方案)
2. **整合到 orbit-engine/**
3. **獨立新專案**

---

## 方案分析

### 方案 1: 保留在 handover-rl/ ⭐ 推薦

#### 優點
✅ **職責清晰**
- orbit-engine: 衛星池優化（選擇哪些衛星）
- handover-rl: 訓練加速（如何快速訓練）
- 兩者目的不同，職責分離

✅ **依賴方向正確**
```
orbit-engine (基礎設施)
    ↑ 依賴
handover-rl (應用層，包含訓練加速)
```
- handover-rl 依賴 orbit-engine ✅
- orbit-engine 不依賴 handover-rl ✅

✅ **使用場景明確**
- 預計算是**訓練專用**的優化
- 其他使用 orbit-engine 的項目不一定需要預計算
- 避免 orbit-engine 承擔過多職責

✅ **開發靈活性**
- handover-rl 可以獨立迭代預計算策略
- 不影響 orbit-engine 的穩定性
- 可以針對RL訓練特性優化

✅ **代碼組織**
```
handover-rl/
├── src/
│   ├── adapters/
│   │   ├── orbit_engine_adapter.py      # 調用 orbit-engine
│   │   ├── orbit_precompute_generator.py # 使用 adapter 生成表
│   │   ├── orbit_precompute_table.py     # 查表後端
│   │   └── adapter_wrapper.py            # 統一接口
│   ├── environments/                     # 使用 wrapper
│   └── agents/                           # 使用 environment
├── scripts/
│   └── generate_orbit_precompute.py     # 生成工具
└── data/
    └── orbit_precompute_7days.h5        # 預計算表
```

#### 缺點
❌ **可能的代碼重複**
- 如果未來其他RL項目也需要預計算
- 需要複製代碼或發布為library

❌ **orbit-engine升級時需要驗證**
- orbit-engine API變更時需要更新 adapter
- 但這是正常的依賴管理

---

### 方案 2: 整合到 orbit-engine/

#### 優點
✅ **統一的資料處理**
- 所有預處理集中在一起
- Stage 1-4 + Precompute Stage 5?

✅ **代碼重用**
- 其他項目可直接使用預計算功能
- 統一的HDF5格式標準

#### 缺點
❌ **職責混淆**
- orbit-engine 原本是**通用軌道計算引擎**
- 加入訓練專用的優化會模糊邊界
- 違反單一職責原則

❌ **依賴膨脹**
```
orbit-engine 需要增加:
├── h5py (HDF5處理)
├── tqdm (進度條)
├── multiprocessing 優化
└── RL訓練相關的時間窗口邏輯
```
- 使用 orbit-engine 的其他項目可能不需要這些

❌ **耦合增加**
- orbit-engine 需要知道RL訓練的需求
- 修改 orbit-engine 影響所有下游項目
- 降低了模塊化程度

❌ **測試複雜度**
- orbit-engine 需要測試預計算功能
- 測試矩陣變大（物理計算 × 預計算策略）

---

### 方案 3: 獨立新專案 (orbit-precompute)

#### 優點
✅ **完全解耦**
```
orbit-engine (軌道計算)
    ↑
orbit-precompute (預計算中間層)
    ↑
handover-rl (RL訓練)
```

✅ **可重用性**
- 其他RL項目可以使用
- 統一的預計算標準

#### 缺點
❌ **過度設計**
- 當前只有一個使用者 (handover-rl)
- 增加維護負擔（3個倉庫 vs 2個）

❌ **開發效率**
- 需要管理額外的發布週期
- 版本兼容性問題（orbit-engine + orbit-precompute + handover-rl）

❌ **過早抽象**
- YAGNI原則：You Aren't Gonna Need It
- 沒有足夠的使用場景證明需要獨立項目

---

## 決策矩陣

| 標準 | 方案1: handover-rl | 方案2: orbit-engine | 方案3: 獨立項目 |
|------|-------------------|---------------------|----------------|
| **職責清晰** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **依賴管理** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **開發效率** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **可重用性** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **維護成本** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **測試簡單** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **總分** | **28/30** | **17/30** | **21/30** |

---

## 推薦決策：方案1 (保留在 handover-rl) ⭐

### 理由

1. **遵循軟件工程原則**
   - Single Responsibility Principle (SRP)
   - Dependency Inversion Principle (DIP)
   - YAGNI (You Aren't Gonna Need It)

2. **符合實際使用場景**
   - 預計算是**訓練加速的優化手段**
   - 不是通用的軌道計算功能
   - 與RL訓練緊密相關

3. **清晰的架構層次**
   ```
   Layer 3: handover-rl (應用層)
            - RL訓練邏輯
            - 訓練優化（預計算）
            - 評估和分析

   Layer 2: orbit-engine (服務層)
            - 軌道計算
            - 物理模型
            - 數據管道

   Layer 1: 基礎設施
            - SGP4, Skyfield
            - ITU-R, 3GPP標準
   ```

4. **未來擴展路徑**
   - 如果有2-3個項目需要預計算 → 考慮抽取為library
   - 如果只有handover-rl使用 → 保持當前架構
   - 決策可以延後，不影響當前開發

---

## 實施建議

### 當前架構 (保持)

```
handover-rl/
├── src/
│   ├── adapters/
│   │   ├── __init__.py                   # 統一接口
│   │   ├── orbit_engine_adapter.py       # orbit-engine調用層
│   │   ├── orbit_precompute_generator.py # 預計算生成
│   │   ├── orbit_precompute_table.py     # 預計算查詢
│   │   ├── adapter_wrapper.py            # 後端切換
│   │   └── _precompute_worker.py         # 並行計算
│   └── ...
├── scripts/
│   └── generate_orbit_precompute.py      # CLI工具
└── docs/
    ├── PRECOMPUTE_DESIGN.md
    ├── PRECOMPUTE_QUICKSTART.md
    └── TRAINING_GUIDE.md
```

### 文檔說明

在 `README.md` 中明確說明：

```markdown
## Architecture

### handover-rl (This Repository)
- RL algorithms (DQN, DDQN, etc.)
- Training infrastructure
- **Training acceleration** (orbit precompute)
- Evaluation and analysis

### orbit-engine (Dependency)
- Orbital mechanics (SGP4)
- Physical models (ITU-R, 3GPP)
- Satellite pool optimization (Stage 1-4)

### Relationship
handover-rl → uses → orbit-engine
```

### 與 orbit-engine 的接口

保持清晰的接口邊界：

```python
# handover-rl 只使用 orbit-engine 的公開 API
from orbit_engine import OrbitEngine

# handover-rl 內部實現訓練加速
from adapters import AdapterWrapper  # 內部實現
```

---

## 未來考慮

### 何時應該獨立？

**條件** (同時滿足3個):
1. 有 **2-3個獨立項目** 需要預計算功能
2. 預計算邏輯已經**穩定** (API不常變動)
3. 有**資源**維護獨立項目

**評估時機**:
- 每6個月評估一次
- 或當新項目需要使用時

### 如何平滑遷移？

如果未來決定獨立：

```
步驟1: 創建 orbit-precompute 倉庫
步驟2: 移動代碼
  handover-rl/src/adapters/orbit_precompute_*
  → orbit-precompute/src/
步驟3: 發布為 pip package
步驟4: handover-rl 改為依賴
  pip install orbit-precompute
步驟5: 更新文檔
```

影響：最小（已經是獨立模塊）

---

## 結論

**決策**: 保留預計算系統在 `handover-rl/` ⭐

**原因**:
1. 職責清晰（訓練加速 vs 軌道計算）
2. 依賴方向正確（應用層 → 服務層）
3. 開發效率高（單一倉庫）
4. 未來可平滑遷移（模塊化設計）

**行動**:
- ✅ 保持當前架構
- ✅ 在文檔中明確說明職責劃分
- ⏳ 每6個月評估一次是否需要獨立

---

**審查日期**: 2025-11-08
**下次審查**: 2026-05-08
**決策者**: Architecture Review
**狀態**: ✅ Approved
