# handover-rl 目錄清理計畫

## 📊 現況分析

### 問題
- 41個 log 文件（總計 ~45MB）
- 30+ markdown 文檔（很多重複/過時）
- 20+ shell 腳本（部分已不使用）
- 2個奇怪的文件：`=0.12.0`, `=2.0.0`
- 根目錄混亂，難以找到重要文件

### 目標
- ✅ 清晰的目錄結構
- ✅ 保留重要的訓練結果
- ✅ 歸檔過時但可能需要參考的文件
- ✅ 刪除真正無用的文件
- ✅ 減少根目錄文件數量

---

## 🗂️ 新目錄結構

```
handover-rl/
├── README.md                          # 專案說明
├── requirements.txt                   # 依賴
├── setup_env.sh                       # 環境設置
├── train.py                          # 主訓練腳本
├── evaluate.py                       # 評估腳本
│
├── src/                              # 源代碼（保持不變）
├── config/                           # 配置（保持不變）
├── scripts/                          # 腳本（保持不變）
├── data/                             # 數據（保持不變）
├── tests/                            # 測試（保持不變）
│
├── checkpoints/                      # 模型檢查點（保持不變）
├── figures/                          # 論文圖表（保持不變）
├── tables/                           # 論文表格（保持不變）
│
├── logs/                             # 🆕 當前訓練記錄
│   ├── training_level5_20min_final.log    # 最終訓練
│   ├── training_monitor.log               # 監控記錄
│   └── figure_generation.log              # 圖表生成
│
├── tools/                            # 🆕 常用工具腳本
│   ├── check_progress.sh                  # 查看進度
│   ├── view_training_log.sh               # 查看日誌
│   ├── generate_paper_figures.sh          # 生成圖表
│   └── auto_monitor.sh                    # 自動監控
│
├── docs/                             # 🆕 重要文檔（整合）
│   ├── ACADEMIC_ACCELERATION_PLAN.md      # 加速計畫（最新）
│   ├── PAPER_FIGURES_GUIDE.md             # 論文圖表指南
│   └── README.md                          # 文檔索引
│
└── archive/                          # 🆕 歸檔（舊文件）
    ├── logs/                         # 舊訓練記錄
    │   ├── level1-4/                 # Level 1-4 實驗
    │   ├── diagnostics/              # 診斷測試
    │   ├── tests/                    # 測試記錄
    │   └── evaluations/              # 評估記錄
    │
    ├── docs/                         # 過時文檔
    │   ├── guides/                   # 各種指南
    │   └── reports/                  # 舊報告
    │
    └── scripts/                      # 過時腳本
        ├── monitoring/               # 舊監控腳本
        └── testing/                  # 舊測試腳本
```

---

## 📝 詳細清理動作

### 1. Log 文件分類

#### 1.1 保留在 `logs/` (3個)
```
✅ training_level5_20min_final.log        # 最終訓練結果
✅ training_monitor.log                   # 監控記錄
✅ figure_generation.log                  # 圖表生成
```

#### 1.2 歸檔到 `archive/logs/level1-4/` (11個)
```
📦 training_level2.log
📦 training_level2_adjusted.log
📦 training_level2_multi_obj.log
📦 training_level2_multi_obj_fixed.log
📦 training_level2_reward_adjusted.log
📦 training_level2_stabilized.log
📦 training_level3.log
📦 training_level3_stable.log
📦 training_level4_official.log
📦 level3_monitor.log
📦 level4_monitor.log
```

#### 1.3 歸檔到 `archive/logs/diagnostics/` (3個)
```
📦 diagnostic_level1.log
📦 diagnostic_level4_test1.log
📦 diagnostic_level5_test1.log
```

#### 1.4 歸檔到 `archive/logs/tests/` (8個)
```
📦 test_20min_config.log
📦 test_full_episodes_30cores.log
📦 test_full_episodes.log
📦 test_full_episodes_v2.log
📦 test_multicore_30.log
📦 training_epsilon_fix.log
📦 training_vanilla_conservative.log
📦 training_ddqn_test_2000.log
```

#### 1.5 歸檔到 `archive/logs/evaluations/` (7個)
```
📦 evaluation.log
📦 evaluation_fixed.log
📦 evaluation_full.log
📦 evaluation_level2.log
📦 evaluation_level3.log
📦 evaluation_level3_fixed.log
📦 evaluation_level4.log
```

#### 1.6 刪除（無用/過時）(3個)
```
🗑️ training.log                          # 空或過時
🗑️ training_new.log                      # 實驗性，已被取代
🗑️ training_level5_20min_final.log.INVALID_ACTIONS_20251103_161704  # 備份，已修復
```

---

### 2. Markdown 文檔整理

#### 2.1 保留在 `docs/` (5個 - 重要文檔)
```
✅ ACADEMIC_ACCELERATION_PLAN.md          # 最新加速計畫
✅ PAPER_FIGURES_GUIDE.md                 # 論文圖表指南
✅ INTEGRATION_GUIDE.md                   # 系統整合指南
✅ README.md                              # 專案說明（移到 docs/）
✅ temp.md                                # 臨時筆記（保留在根目錄）
```

#### 2.2 歸檔到 `archive/docs/guides/` (13個)
```
📦 MONITORING_GUIDE.md
📦 REALTIME_MONITORING_GUIDE.md
📦 VISUALIZATION_GUIDE.md
📦 FIGURES_QUICK_REFERENCE.md
📦 PAPER_FIGURES_SUMMARY.md
📦 REALTIME_SYSTEM_SUMMARY.md
📦 MULTICORE_STATUS.md
📦 MULTICORE_TRAINING.md
📦 TRAINING_PLAN.md
📦 ENVIRONMENT_MIGRATION_CHECKLIST.md
📦 ACADEMIC_COMPLIANCE.md
📦 ACADEMIC_COMPLIANCE_REPORT.md
📦 REFERENCES.md
```

#### 2.3 歸檔到 `archive/docs/reports/` (6個)
```
📦 VERIFICATION_REPORT.md
📦 SUMMARY.md
📦 level_verification.md
📦 time_analysis.md
📦 multicore_analysis.md
📦 final_recommendation.md
📦 PARALLEL_TASKS.md
```

#### 2.4 刪除（已合併到新計畫）(2個)
```
🗑️ TODO.md                               # 已完成，內容已整合
🗑️ CHANGELOG.md                          # 可以從 git 歷史查看
```

#### 2.5 保留在根目錄
```
✅ temp.md                                # 工作筆記
✅ temp_backup.md                         # 備份
```

---

### 3. Shell 腳本整理

#### 3.1 移動到 `tools/` (7個 - 常用工具)
```
✅ check_progress.sh                      # 查看進度
✅ view_training_log.sh                   # 查看日誌
✅ view_monitor.sh                        # 查看監控
✅ generate_paper_figures.sh              # 生成圖表
✅ auto_monitor.sh                        # 自動監控
✅ analyze_training.sh                    # 分析訓練
✅ train_level5_final.sh                  # 最終訓練腳本
```

#### 3.2 歸檔到 `archive/scripts/monitoring/` (8個)
```
📦 start_monitor.sh
📦 monitor_training.sh
📦 monitor_level3.sh
📦 monitor_level4.sh
📦 monitor_level5.sh
📦 monitor_episode920.sh
📦 monitor_30cores.sh
📦 notify_milestones.sh
📦 dashboard.sh
```

#### 3.3 歸檔到 `archive/scripts/testing/` (4個)
```
📦 test_20min_config.sh
📦 test_multicore.sh
📦 run_test_30cores.sh
📦 quick_check.sh
📦 quick_train.sh
```

---

### 4. 其他文件處理

#### 4.1 刪除（垃圾文件）
```
🗑️ =0.12.0                               # pip 安裝錯誤產生
🗑️ =2.0.0                                # pip 安裝錯誤產生
```

#### 4.2 保留
```
✅ test_action_masking.py                 # 驗證腳本（移到 tests/）
✅ train_offline_bc_v4_candidate_pool.py  # 訓練腳本（保留根目錄）
✅ train_online_rl.py                     # 訓練腳本（保留根目錄）
✅ live_monitor.html                      # 監控頁面（移到 tools/）
✅ training_milestones.txt                # 里程碑記錄（移到 logs/）
```

#### 4.3 目錄保持不變
```
✅ src/
✅ config/
✅ scripts/
✅ data/
✅ tests/
✅ checkpoints/
✅ figures/
✅ tables/
✅ venv/
✅ __pycache__/
✅ archive/                               # 已存在
✅ output/
✅ frontend/
✅ api/
✅ docker-compose.yml
✅ Dockerfile
```

---

## 📊 清理效果預估

### 根目錄文件數量
- **清理前**: ~70個文件
- **清理後**: ~15個核心文件
- **減少**: 78%

### 磁盤空間
- **Log 歸檔**: ~45MB → 保留 ~400KB
- **文檔整理**: 更清晰的結構
- **總體**: 不刪除重要數據，只是重新組織

### 可維護性
- ✅ 清晰的目錄結構
- ✅ 快速找到重要文件
- ✅ 歷史記錄保存在 archive
- ✅ 新實驗不會再混亂

---

## 🚀 執行步驟

### Step 1: 創建新目錄結構
```bash
mkdir -p logs
mkdir -p tools
mkdir -p docs
mkdir -p archive/{logs,docs,scripts}/{level1-4,diagnostics,tests,evaluations,guides,reports,monitoring,testing}
```

### Step 2: 移動 Log 文件
```bash
# 保留當前 logs
mv training_level5_20min_final.log logs/
mv training_monitor.log logs/
mv figure_generation.log logs/
mv training_milestones.txt logs/

# 歸檔舊 logs
mv training_level{2,3,4}*.log archive/logs/level1-4/
mv level{3,4}_monitor.log archive/logs/level1-4/
mv diagnostic_*.log archive/logs/diagnostics/
mv test_*.log archive/logs/tests/
mv training_{epsilon_fix,vanilla_conservative,ddqn_test_2000}.log archive/logs/tests/
mv evaluation*.log archive/logs/evaluations/

# 刪除無用
rm -f training.log training_new.log
rm -f training_level5_20min_final.log.INVALID_ACTIONS_*
```

### Step 3: 整理文檔
```bash
# 移動到 docs/
mv ACADEMIC_ACCELERATION_PLAN.md docs/
mv PAPER_FIGURES_GUIDE.md docs/
mv INTEGRATION_GUIDE.md docs/

# 歸檔舊文檔
mv *_GUIDE.md archive/docs/guides/
mv *_SUMMARY.md archive/docs/guides/
mv *_STATUS.md archive/docs/guides/
mv MULTICORE_*.md archive/docs/guides/
mv ACADEMIC_COMPLIANCE*.md archive/docs/guides/
mv ENVIRONMENT_*.md archive/docs/guides/
mv TRAINING_PLAN.md archive/docs/guides/
mv REFERENCES.md archive/docs/guides/

mv VERIFICATION_REPORT.md archive/docs/reports/
mv SUMMARY.md archive/docs/reports/
mv level_verification.md archive/docs/reports/
mv time_analysis.md archive/docs/reports/
mv multicore_analysis.md archive/docs/reports/
mv final_recommendation.md archive/docs/reports/
mv PARALLEL_TASKS.md archive/docs/reports/

# 刪除
rm -f TODO.md CHANGELOG.md
```

### Step 4: 整理腳本
```bash
# 移動到 tools/
mv check_progress.sh tools/
mv view_training_log.sh tools/
mv view_monitor.sh tools/
mv generate_paper_figures.sh tools/
mv auto_monitor.sh tools/
mv analyze_training.sh tools/
mv train_level5_final.sh tools/
mv live_monitor.html tools/

# 歸檔舊腳本
mv monitor_*.sh archive/scripts/monitoring/
mv start_monitor.sh archive/scripts/monitoring/
mv notify_milestones.sh archive/scripts/monitoring/
mv dashboard.sh archive/scripts/monitoring/

mv test_*.sh archive/scripts/testing/
mv run_test_*.sh archive/scripts/testing/
mv quick_*.sh archive/scripts/testing/
```

### Step 5: 其他文件
```bash
# 刪除垃圾
rm -f =0.12.0 =2.0.0

# 移動測試腳本
mv test_action_masking.py tests/
```

### Step 6: 創建索引文檔
```bash
# 在 docs/ 創建 README
# 在 archive/ 創建 README
# 在 tools/ 創建 README
```

---

## ⚠️ 注意事項

1. **備份**: 執行前先 git commit 當前狀態
2. **路徑更新**: 某些腳本可能引用了舊路徑，需要更新
3. **測試**: 清理後測試關鍵功能（訓練、評估、圖表生成）
4. **文檔**: 更新 README 說明新的目錄結構

---

## ✅ 驗證清單

清理完成後檢查：
- [ ] 根目錄只有核心文件（<20個）
- [ ] logs/ 包含最新訓練記錄
- [ ] tools/ 包含常用工具腳本
- [ ] docs/ 包含重要文檔
- [ ] archive/ 包含所有歷史文件
- [ ] 訓練腳本仍可正常運行
- [ ] 圖表生成腳本仍可正常運行
- [ ] 無重要文件丟失
