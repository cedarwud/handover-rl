# 歷史文件歸檔

本目錄保存所有歷史文件，用於參考和審查。

⚠️ **注意**: 此目錄中的文件是歷史記錄，不應用於當前開發。

---

## 📁 目錄結構

```
archive/
├── logs/              # 歷史訓練日誌
│   ├── level1-4/      # Level 1-4 訓練記錄
│   ├── diagnostics/   # 診斷測試記錄
│   ├── tests/         # 各種測試記錄
│   └── evaluations/   # 評估記錄
│
├── docs/              # 歷史文檔
│   ├── guides/        # 各種操作指南
│   └── reports/       # 歷史報告
│
└── scripts/           # 歷史腳本
    ├── monitoring/    # 舊監控腳本
    └── testing/       # 舊測試腳本
```

---

## 📊 logs/ - 歷史訓練日誌

### level1-4/ (11個文件, ~15MB)
Level 1-4 數值穩定性實驗的訓練記錄：
- `training_level2*.log`: Level 2 實驗（獎勵調整、多目標等）
- `training_level3*.log`: Level 3 實驗
- `training_level4_official.log`: Level 4 正式訓練
- `level3_monitor.log`, `level4_monitor.log`: 監控記錄

### diagnostics/ (3個文件, ~5.5MB)
數值穩定性診斷測試：
- `diagnostic_level1.log`: Level 1 診斷
- `diagnostic_level4_test1.log`: Level 4 診斷
- `diagnostic_level5_test1.log`: Level 5 診斷

### tests/ (8個文件, ~30MB)
各種測試和實驗性訓練：
- `test_20min_config.log`: 20分鐘 episode 配置測試
- `test_full_episodes_30cores.log`: 30核心多進程測試（最大，20MB）
- `test_multicore_30.log`: 多核訓練測試
- `training_epsilon_fix.log`: Epsilon 修復測試
- `training_vanilla_conservative.log`: 保守訓練測試
- `training_ddqn_test_2000.log`: Double DQN 測試

### evaluations/ (7個文件, ~400KB)
模型評估記錄：
- `evaluation*.log`: 各個版本的評估結果

---

## 📚 docs/ - 歷史文檔

### guides/ (13個文件)
操作指南和系統說明：
- `MONITORING_GUIDE.md`: 監控系統指南
- `REALTIME_MONITORING_GUIDE.md`: 實時監控指南
- `VISUALIZATION_GUIDE.md`: 視覺化指南
- `FIGURES_QUICK_REFERENCE.md`: 圖表快速參考
- `PAPER_FIGURES_SUMMARY.md`: 論文圖表總結
- `MULTICORE_*.md`: 多核訓練相關文檔
- `TRAINING_PLAN.md`: 訓練計畫
- `ACADEMIC_COMPLIANCE*.md`: 學術標準文檔
- `ENVIRONMENT_MIGRATION_CHECKLIST.md`: 環境遷移檢查清單
- `REFERENCES.md`: 參考文獻

### reports/ (6個文件)
歷史分析報告：
- `VERIFICATION_REPORT.md`: 驗證報告
- `SUMMARY.md`: 總結報告
- `level_verification.md`: Level 驗證
- `time_analysis.md`: 時間分析
- `multicore_analysis.md`: 多核分析
- `final_recommendation.md`: 最終建議
- `PARALLEL_TASKS.md`: 並行任務

---

## 🔧 scripts/ - 歷史腳本

### monitoring/ (9個文件)
舊監控腳本（已被 `../tools/auto_monitor.sh` 取代）：
- `monitor_level3.sh`, `monitor_level4.sh`, `monitor_level5.sh`
- `monitor_episode920.sh`: Episode 920 專用監控
- `monitor_30cores.sh`: 30核心監控
- `start_monitor.sh`, `monitor_training.sh`
- `notify_milestones.sh`: 里程碑通知
- `dashboard.sh`: 監控儀表板

### testing/ (4個文件)
舊測試腳本：
- `test_20min_config.sh`: 配置測試
- `test_multicore.sh`: 多核測試
- `run_test_30cores.sh`: 30核心測試
- `quick_check.sh`, `quick_train.sh`: 快速測試工具

---

## 🔍 何時查看歷史文件

### 需要參考時
- 想了解過去的實驗配置
- 對比不同 Level 的訓練結果
- 查看問題的歷史解決方案
- 研究多核訓練的嘗試

### 不建議使用
- ❌ 直接運行歷史腳本（可能已過時）
- ❌ 使用歷史配置（可能有已知問題）
- ❌ 參考過時的指南（優先看 `../docs/`）

---

## 📌 當前文件位置

| 類型 | 當前位置 |
|------|----------|
| 最新訓練日誌 | `../logs/` |
| 重要文檔 | `../docs/` |
| 常用工具 | `../tools/` |
| 論文圖表 | `../figures/` |
| 檢查點 | `../checkpoints/` |

---

**歸檔時間**: 2025-11-08
**歸檔原因**: 目錄整理，保持根目錄清潔
