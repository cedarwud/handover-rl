# Archive 目錄

此目錄包含已完成任務的臨時文件和過時文檔，保留作為歷史記錄參考。

**歸檔日期**: 2025-10-21

---

## 📁 目錄結構

```
archive/
├── scripts/     # 臨時腳本（已完成任務）
├── docs/        # 過時文檔
└── logs/        # 臨時日誌文件
```

---

## 📝 Scripts（腳本）

### BC 訓練腳本演進歷史

| 文件 | 版本 | 狀態 | 說明 |
|------|------|------|------|
| `train_offline_bc.py` | V1 | ❌ 失敗 | 初版，100% 準確率（數據洩漏） |
| `train_offline_bc_v2.py` | V2 | ❌ 失敗 | 改進嘗試，仍有洩漏 |
| `train_offline_bc_v3.py` | V3 | ❌ 失敗 | Negative sampling 錯誤 |
| `train_offline_bc_v5_final.py` | V5 | ❌ 失敗 | Timestamp mismatch (Stage 5/6 不同日期) |

**✅ 最終成功版本**: `train_offline_bc_v4_candidate_pool.py` (根目錄)
- 準確率: 88.81%
- 數據洩漏: 已完全消除
- 學習曲線: 平滑健康

### 分析腳本（已完成）

| 文件 | 用途 | 產出 |
|------|------|------|
| `analyze_actual_handover_events.py` | 分析 48,002 真實換手事件 | 閾值建議 |
| `analyze_dataset.py` | 數據集統計 | 數據品質報告 |
| `analyze_real_data_for_thresholds.py` | 數據驅動閾值設計 | FINAL_THRESHOLD_RECOMMENDATIONS.md |
| `analyze_training_features.py` | 特徵分析 | 訓練特徵報告 |
| `diagnose_data_leakage.py` | 診斷 100% 準確率問題 | DIAGNOSIS_100_ACCURACY.md |

### 驗證腳本（已完成）

| 文件 | 用途 | 結果 |
|------|------|------|
| `verify_a4_threshold.py` | 驗證 A4 閾值修復 | ✅ -34.5 dBm |
| `verify_all_thresholds.py` | 驗證所有閾值 | ✅ 全部正確 |
| `verify_new_thresholds.py` | 驗證新閾值效果 | ✅ Margin 2-15 dB |

### 其他腳本

- `test_refactored_framework.py` - 重構框架測試（已通過）

---

## 📚 Docs（文檔）

### 階段完成報告（已過時）

| 文件 | 階段 | 完成日期 |
|------|------|----------|
| `PHASE1_COMPLETE.md` | Phase 1 完成 | 2025-10-20 |
| `PHASE1_COMPLETION_SUMMARY.md` | Phase 1 總結 | 2025-10-20 |
| `PHASE2_COMPLETE.md` | Phase 2 完成 | 2025-10-20 |
| `PHASE2_COMPLETION_SUMMARY.md` | Phase 2 總結 | 2025-10-20 |
| `PROJECT_COMPLETE.md` | 項目完成聲明 | 2025-10-20 |
| `REFACTORING_COMPLETE.md` | 重構完成 | 2025-10-20 |

**為什麼過時**:
- 這些文檔是在 BC 訓練達到 100% 準確率時寫的
- 後來發現 100% 是數據洩漏問題，並非成功
- 真正的完成報告在根目錄的 `FINAL_SOLUTION_SUMMARY.md`

### 閾值建議（已被取代）

| 文件 | 取代者 |
|------|--------|
| `threshold_recommendations.md` | `FINAL_THRESHOLD_RECOMMENDATIONS.md` |

---

## 📋 Logs（日誌）

| 文件 | 內容 |
|------|------|
| `actual_handover_analysis.log` | 換手事件分析日誌 |
| `bc_training_smoke_test.log` | BC 訓練測試日誌 |
| `bc_training_v2.log` | V2 訓練日誌 |
| `bc_training_v3.log` | V3 訓練日誌 |
| `stage6_rerun.log` | Stage 6 重新執行日誌 |
| `threshold_analysis_real_data.log` | 閾值分析日誌 |
| `rl_training_generation.log` | RL 數據生成日誌 |

**最新日誌位置**: `/tmp/bc_training_v4_with_checkpoints.log`

---

## 🗑️ 已刪除文件

| 文件 | 原因 |
|------|------|
| `backup_offline_rl_20251019.tar.gz` | 舊備份，已被最新模型取代 |

---

## 📌 參考價值

這些歷史文件對以下情況有參考價值：

1. **論文撰寫** - 展示問題診斷和解決過程
2. **錯誤學習** - 了解為什麼某些方法失敗
3. **方法論** - 記錄實驗演進過程

---

## ✅ 當前有效文件

**根目錄的核心文件**（不在 archive 中）:

### 訓練腳本
- `train.py` - DQN 主訓練入口 ⭐
- `train_online_rl.py` - 線上 RL 訓練
- `train_offline_bc_v4_candidate_pool.py` - BC V4 訓練（成功版本）✅

### 文檔
- `TODO.md` - 項目狀態與待辦事項 ⭐
- `FINAL_SOLUTION_SUMMARY.md` - 完整解決方案總結
- `TRAINING_REPORT_V4_FINAL.md` - BC V4 訓練報告
- `DIAGNOSIS_100_ACCURACY.md` - 數據洩漏診斷
- `FINAL_THRESHOLD_RECOMMENDATIONS.md` - 閾值建議
- `README.md` - 項目說明
- `CHANGELOG.md` - 變更記錄

### 配置
- `requirements.txt` - Python 依賴
- `setup_env.sh` - 環境設置
- `quick_train.sh` - 快速訓練

### 模型
- `checkpoints/bc_policy_v4_best_20251021_020013.pth` - 最佳 BC 模型 ⭐

---

**歸檔者**: Claude Code
**歸檔日期**: 2025-10-21 02:15:00
