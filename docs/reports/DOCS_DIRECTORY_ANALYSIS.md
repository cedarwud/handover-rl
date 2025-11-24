# docs/ 目錄深度分析報告

**分析日期**: 2024-11-24
**問題**: docs/ 中有 8 個文檔，是否都需要？如何組織？

---

## 📊 當前 docs/ 目錄結構

```
docs/
├── ACADEMIC_ACCELERATION_PLAN.md          (438 lines, 11 KB)
├── ACADEMIC_COMPLIANCE_CHECKLIST.md       (366 lines, 9 KB)
├── INTEGRATION_GUIDE.md                   (609 lines, 13 KB)
├── PAPER_FIGURES_GUIDE.md                 (657 lines, 16 KB)
├── PRECOMPUTE_ARCHITECTURE_DECISION.md    (307 lines, 7.6 KB)
├── PRECOMPUTE_DESIGN.md                   (452 lines, 14 KB)
├── PRECOMPUTE_QUICKSTART.md               (292 lines, 6.9 KB)
├── TRAINING_GUIDE.md                      (464 lines, 11 KB)
└── reports/                               (26 files, ~300 KB)
```

**總計**: 8 個主要文檔 + 26 個報告

---

## 🔍 每個文檔的分析

### 1. TRAINING_GUIDE.md ✅ **必須保留**

**內容**: 多級訓練策略（Level 0-6）
**用途**: 用戶訓練的主要指南
**受眾**: 所有用戶
**狀態**: ✅ 已更新 (config/ → configs/)
**評分**: 10/10 (核心文檔)

**結論**: ✅ **保留在 docs/**

---

### 2. PRECOMPUTE_QUICKSTART.md ✅ **必須保留**

**內容**: 預計算系統快速開始
**用途**: 用戶設置預計算表的指南
**受眾**: 所有用戶
**狀態**: ✅ 已更新 (config/ → configs/)
**評分**: 10/10 (核心文檔)

**結論**: ✅ **保留在 docs/**

---

### 3. PRECOMPUTE_DESIGN.md ✅ **必須保留**

**內容**: 預計算系統技術設計
**用途**: 理解系統架構
**受眾**: 開發者、研究者
**狀態**: ✅ 已更新 (config/ → configs/)
**評分**: 9/10 (重要文檔)

**結論**: ✅ **保留在 docs/**

---

### 4. ACADEMIC_COMPLIANCE_CHECKLIST.md ✅ **必須保留**

**內容**: 學術標準驗證清單
**用途**: 驗證系統符合學術標準
**受眾**: 研究者、審稿人
**狀態**: ✅ 已更新 (config/ → configs/)
**評分**: 10/10 (學術必需)

**結論**: ✅ **保留在 docs/**

---

### 5. PRECOMPUTE_ARCHITECTURE_DECISION.md ✅ **保留**

**內容**: 為什麼在 handover-rl 而不是 orbit-engine 實現預計算
**用途**: 架構決策文檔
**受眾**: 開發者、架構師
**狀態**: ✅ 已更新 (config/ → configs/)
**評分**: 8/10 (重要但不常用)

**結論**: ✅ **保留在 docs/** (但可考慮移至 docs/design/)

---

### 6. PAPER_FIGURES_GUIDE.md ⚠️ **評估**

**內容**: 論文圖表生成指南（657 lines）
**用途**: 生成和使用論文圖表
**受眾**: 論文撰寫者
**狀態**: 未檢查 config/ 引用
**評分**: 7/10 (專用文檔)

**分析**:
- ✅ 內容有用（圖表生成、LaTeX 集成）
- ⚠️ 內容非常長 (657 lines)
- ⚠️ 較專業（只有寫論文時需要）

**選項**:
1. **保留在 docs/** - 如果是常用指南
2. **移至 docs/paper/** - 如果是專用指南
3. **移至 results/figures/README.md** - 如果主要是圖表說明

**建議**: **移至 docs/paper/FIGURES_GUIDE.md** (創建 paper 子目錄)

---

### 7. INTEGRATION_GUIDE.md ⚠️ **評估**

**內容**: 訓練監控系統整合指南（609 lines）
**用途**: 將監控系統整合到 leo-simulator 和 orbit-engine
**受眾**: 系統集成人員
**狀態**: 未檢查 config/ 引用
**評分**: 6/10 (較舊，可能過時)

**分析**:
- ⚠️ 內容提到 "leo-simulator (前端)" 和 "orbit-engine (後端)"
- ⚠️ 但我們的前端在 tools/frontend/
- ⚠️ API 在 tools/api/
- ❓ 這個指南是否還適用？

**日期**: 2024-11-03 (在重組之前)

**選項**:
1. **更新並保留** - 如果內容仍然相關
2. **移至 docs/legacy/** - 如果已過時
3. **移至 tools/README.md** - 如果是工具專用指南

**建議**: **檢查內容相關性，可能移至 docs/integration/ 或 tools/**

---

### 8. ACADEMIC_ACCELERATION_PLAN.md ⚠️ **評估**

**內容**: 學術標準訓練加速計畫（438 lines）
**用途**: 30天完成發表級研究的路線圖
**受眾**: 研究計劃人員
**狀態**: 未檢查 config/ 引用
**評分**: 7/10 (計劃文檔)

**分析**:
- ⚠️ 這是一個"計劃"文檔
- ⚠️ 但訓練已經完成（Level 5, 6）
- ❓ 這個計劃是否還需要？
- ❓ 或者應該作為"歷史記錄"保存？

**日期**: 2024-11-08

**選項**:
1. **保留在 docs/** - 如果未來還需要參考
2. **移至 docs/planning/** - 如果是計劃類文檔
3. **移至 docs/reports/** - 如果是歷史記錄
4. **移至 archive/docs/** - 如果已完成不再需要

**建議**: **移至 docs/planning/ACCELERATION_PLAN.md** 或 **docs/reports/**

---

## 📋 推薦的 docs/ 目錄重組

### 選項 A: 扁平結構（當前）

```
docs/
├── TRAINING_GUIDE.md                      ✅ 核心
├── PRECOMPUTE_QUICKSTART.md               ✅ 核心
├── PRECOMPUTE_DESIGN.md                   ✅ 核心
├── PRECOMPUTE_ARCHITECTURE_DECISION.md    ✅ 核心
├── ACADEMIC_COMPLIANCE_CHECKLIST.md       ✅ 核心
├── PAPER_FIGURES_GUIDE.md                 ⚠️ 可移動
├── INTEGRATION_GUIDE.md                   ⚠️ 可移動
├── ACADEMIC_ACCELERATION_PLAN.md          ⚠️ 可移動
└── reports/                               ✅ (26 報告)
```

**優點**: 簡單，所有文檔在同一層
**缺點**: 不同類型文檔混在一起

---

### 選項 B: 分類結構（推薦）⭐

```
docs/
├── 📚 核心文檔（用戶常用）
│   ├── TRAINING_GUIDE.md                  ✅ 訓練指南
│   ├── PRECOMPUTE_QUICKSTART.md           ✅ 快速開始
│   └── ACADEMIC_COMPLIANCE_CHECKLIST.md   ✅ 學術標準
│
├── 🔬 設計文檔（開發者）
│   ├── design/
│   │   ├── PRECOMPUTE_DESIGN.md
│   │   └── PRECOMPUTE_ARCHITECTURE_DECISION.md
│
├── 📊 論文相關（撰寫論文時）
│   └── paper/
│       └── FIGURES_GUIDE.md              ← 移動自 PAPER_FIGURES_GUIDE.md
│
├── 🔗 系統集成（集成人員）
│   └── integration/
│       └── MONITORING_INTEGRATION.md     ← 移動自 INTEGRATION_GUIDE.md
│
├── 📅 計劃與歷史（參考）
│   └── planning/
│       └── ACCELERATION_PLAN.md          ← 移動自 ACADEMIC_ACCELERATION_PLAN.md
│
└── 📋 分析報告（清理過程）
    └── reports/                          ✅ (26 報告)
```

**優點**:
- 清晰分類，易於查找
- 常用文檔在頂層
- 專用文檔在子目錄

**缺點**:
- 目錄層級增加
- 需要更新 README.md 中的連結

---

### 選項 C: 最小化結構（激進）⭐⭐

```
docs/
├── TRAINING_GUIDE.md                      ✅ 保留
├── PRECOMPUTE_QUICKSTART.md               ✅ 保留
├── PRECOMPUTE_DESIGN.md                   ✅ 保留
├── ACADEMIC_COMPLIANCE_CHECKLIST.md       ✅ 保留
├── PRECOMPUTE_ARCHITECTURE_DECISION.md    ✅ 保留
└── reports/                               ✅ 保留
    ├── ... (26 清理報告)
    ├── PAPER_FIGURES_GUIDE.md            ← 移動（已完成，歷史記錄）
    ├── INTEGRATION_GUIDE.md              ← 移動（可能過時）
    └── ACCELERATION_PLAN.md              ← 移動（已完成）
```

**理由**:
- **PAPER_FIGURES_GUIDE.md**: 圖表已生成，這是"如何生成"的記錄
- **INTEGRATION_GUIDE.md**: 系統已整合，這是"如何整合"的記錄
- **ACCELERATION_PLAN.md**: 計劃已完成，這是"計劃過程"的記錄

**優點**:
- docs/ 只保留**當前需要**的文檔
- 歷史和過程文檔在 reports/

**缺點**:
- 如果未來需要重新生成圖表或重新整合，需要去 reports/ 找

---

## 🎯 推薦方案

### 方案 1: 最小化清理（推薦）⭐⭐⭐

**移動 3 個文檔到 docs/reports/**:

```bash
# 1. 論文圖表指南（圖表已生成，作為記錄保存）
mv docs/PAPER_FIGURES_GUIDE.md docs/reports/

# 2. 整合指南（系統已整合，作為記錄保存）
mv docs/INTEGRATION_GUIDE.md docs/reports/

# 3. 加速計劃（計劃已完成，作為記錄保存）
mv docs/ACADEMIC_ACCELERATION_PLAN.md docs/reports/
```

**最終 docs/ 結構** (5 個核心文檔):
```
docs/
├── TRAINING_GUIDE.md                      ✅ 訓練指南
├── PRECOMPUTE_QUICKSTART.md               ✅ 快速開始
├── PRECOMPUTE_DESIGN.md                   ✅ 系統設計
├── PRECOMPUTE_ARCHITECTURE_DECISION.md    ✅ 架構決策
├── ACADEMIC_COMPLIANCE_CHECKLIST.md       ✅ 學術標準
└── reports/                               (29 報告)
    ├── ... (26 清理報告)
    ├── PAPER_FIGURES_GUIDE.md            ← 新增
    ├── INTEGRATION_GUIDE.md              ← 新增
    └── ACCELERATION_PLAN.md              ← 新增
```

**評分**: 9/10

---

### 方案 2: 分類結構（替代）⭐⭐

**創建子目錄**:

```bash
mkdir -p docs/design docs/paper docs/planning

# 移動設計文檔
mv docs/PRECOMPUTE_ARCHITECTURE_DECISION.md docs/design/

# 移動論文文檔
mv docs/PAPER_FIGURES_GUIDE.md docs/paper/

# 移動計劃文檔
mv docs/ACADEMIC_ACCELERATION_PLAN.md docs/planning/

# 移動（或刪除）整合指南
mv docs/INTEGRATION_GUIDE.md docs/reports/  # 或 archive/
```

**最終結構**:
```
docs/
├── TRAINING_GUIDE.md
├── PRECOMPUTE_QUICKSTART.md
├── PRECOMPUTE_DESIGN.md
├── ACADEMIC_COMPLIANCE_CHECKLIST.md
├── design/
│   └── PRECOMPUTE_ARCHITECTURE_DECISION.md
├── paper/
│   └── PAPER_FIGURES_GUIDE.md
├── planning/
│   └── ACADEMIC_ACCELERATION_PLAN.md
└── reports/ (26+)
```

**評分**: 8/10 (更多子目錄，可能過度組織)

---

## 📊 文檔必要性評估

| 文檔 | 當前狀態 | 未來使用 | 推薦操作 |
|------|----------|----------|----------|
| **TRAINING_GUIDE.md** | ✅ 常用 | ✅ 持續需要 | **保留** |
| **PRECOMPUTE_QUICKSTART.md** | ✅ 常用 | ✅ 持續需要 | **保留** |
| **PRECOMPUTE_DESIGN.md** | ✅ 重要 | ✅ 持續需要 | **保留** |
| **ACADEMIC_COMPLIANCE_CHECKLIST.md** | ✅ 重要 | ✅ 持續需要 | **保留** |
| **PRECOMPUTE_ARCHITECTURE_DECISION.md** | ⚠️ 參考 | ⚠️ 偶爾需要 | **保留** |
| **PAPER_FIGURES_GUIDE.md** | ⚠️ 參考 | ❓ 可能需要 | **移至 reports/** |
| **INTEGRATION_GUIDE.md** | ❌ 可能過時 | ❌ 可能不需要 | **移至 reports/** |
| **ACADEMIC_ACCELERATION_PLAN.md** | ❌ 已完成 | ❌ 歷史記錄 | **移至 reports/** |

---

## ✅ 推薦執行步驟

### Step 1: 移動已完成/歷史文檔到 reports/

```bash
mv docs/PAPER_FIGURES_GUIDE.md docs/reports/
mv docs/INTEGRATION_GUIDE.md docs/reports/
mv docs/ACADEMIC_ACCELERATION_PLAN.md docs/reports/
```

### Step 2: 更新 README.md 中的文檔索引

```markdown
### 📖 Core Documentation (5 files)
- docs/TRAINING_GUIDE.md
- docs/PRECOMPUTE_QUICKSTART.md
- docs/PRECOMPUTE_DESIGN.md
- docs/PRECOMPUTE_ARCHITECTURE_DECISION.md
- docs/ACADEMIC_COMPLIANCE_CHECKLIST.md

### 📋 Historical & Reference (in docs/reports/)
- docs/reports/PAPER_FIGURES_GUIDE.md
- docs/reports/INTEGRATION_GUIDE.md
- docs/reports/ACCELERATION_PLAN.md
- docs/reports/... (26 cleanup reports)
```

### Step 3: 檢查並更新所有 3 個移動文檔中的 config/ 引用

```bash
sed -i 's|config/|configs/|g' docs/reports/PAPER_FIGURES_GUIDE.md
sed -i 's|config/|configs/|g' docs/reports/INTEGRATION_GUIDE.md
sed -i 's|config/|configs/|g' docs/reports/ACCELERATION_PLAN.md
```

### Step 4: Git Commit

```bash
git add docs/
git commit -m "docs: Reorganize documentation - move historical docs to reports/

- Move PAPER_FIGURES_GUIDE.md to reports/ (figures already generated)
- Move INTEGRATION_GUIDE.md to reports/ (system already integrated)
- Move ACADEMIC_ACCELERATION_PLAN.md to reports/ (plan completed)
- Keep 5 core docs in docs/ root
- Update all config/ references in moved docs

Result: docs/ now has 5 essential docs + reports/ subdir
"
```

---

## 🎯 最終評估

### 當前 docs/ 狀態: **7/10**
- ✅ 有用的文檔
- ⚠️ 但混雜了歷史/計劃文檔
- ⚠️ 不夠清晰哪些是核心，哪些是參考

### 清理後狀態: **9/10**
- ✅ 5 個核心文檔清晰
- ✅ 歷史/參考文檔在 reports/
- ✅ 易於查找和使用

---

**分析完成日期**: 2024-11-24
**推薦行動**: 執行方案 1（最小化清理）
**預期效果**: docs/ 從 8 個文檔 → 5 個核心文檔
