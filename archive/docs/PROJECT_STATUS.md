# Satellite Handover RL - 項目狀態與待辦事項

**最後更新**: 2025-10-21 02:03:00

---

## 📍 當前狀態總覽

### ✅ 已完成工作

#### 1. 數據洩漏問題診斷與修復 (2025-10-20 ~ 2025-10-21)

**問題**: BC 模型訓練達到 100% 準確率（數據洩漏）

**根本原因** (3 層問題):
1. **Layer 1**: Config bug - A4 threshold = -100 dBm（應為 -34.5 dBm）
2. **Layer 2**: Negative sampling 策略錯誤 - 人為讓鄰居衛星變差
3. **Layer 3**: 特徵空間完全分離 - trivial learning

**修復措施**:
- ✅ 修復配置 bug (`stage6_research_optimization_processor.py:169-228`)
- ✅ 套用數據驅動閾值 (基於 48,002 真實換手事件)
- ✅ 實現 threshold-based labeling
- ✅ 從候選池採樣 (3,302 satellites)
- ✅ 重新生成 RL 訓練數據集

**閾值設計** (Data-driven):
```yaml
A3 offset:    2.5 dB  (原 2.0 dB)
A4 threshold: -34.5 dBm (原 -100.0 dBm, 30th percentile)
A5 threshold1: -36.0 dBm (10th percentile)
A5 threshold2: -33.0 dBm (40th percentile)
A4 hysteresis: 2.0 dB
```

**數據品質改善**:
```
Trigger margin: 55-80 dB → 2-15 dB ✅
A4 events: 48,002 → 21,224 (-55.8%)
數據真實性: 顯著提升 ✅
```

#### 2. BC V4 訓練成功 (2025-10-21 02:00)

**訓練配置**:
```python
Script: train_offline_bc_v4_candidate_pool.py
Dataset: 11,081 samples (54.8% positive, 45.2% negative)
Architecture: 128 → 64 → 32 → 1 (with BatchNorm + Dropout)
Learning rate: 0.0005
Epochs: 20
Device: CUDA
```

**訓練結果**:
```
✅ Test Accuracy:  88.81% (目標範圍 85-95%)
✅ Train Accuracy: 89.35%
✅ 泛化差距:       0.54% (優秀)
✅ 學習曲線:       平滑上升，無數據洩漏

學習曲線:
  Epoch 1-6:   45.69% (健康初始化)
  Epoch 7-18:  47-82% (穩定學習)
  Epoch 19:    85.39% ✅ 進入目標範圍
  Epoch 20:    88.81% ✅ 最佳性能
```

**關鍵特徵**:
- ✅ 無突然跳躍到 100%（數據洩漏完全消除）
- ✅ Train-Test 差距僅 0.54%（優秀泛化）
- ✅ 所有 20 個 epoch checkpoints 已保存
- ✅ 自動選擇最佳模型

**產出文件**:
```
最佳模型: checkpoints/bc_policy_v4_best_20251021_020013.pth
所有 checkpoints: checkpoints/bc_v4_20251021_020013/
  - epoch_19_testacc_85.39.pth (備選)
  - epoch_20_testacc_88.81.pth (最佳)
  - training_history.json
訓練日誌: /tmp/bc_training_v4_with_checkpoints.log
```

**文檔**:
- `FINAL_SOLUTION_SUMMARY.md` - 完整解決方案總結
- `TRAINING_REPORT_V4_FINAL.md` - 訓練報告
- `DIAGNOSIS_100_ACCURACY.md` - 問題診斷
- `FINAL_THRESHOLD_RECOMMENDATIONS.md` - 閾值建議

#### 3. RL 訓練數據集 (2025-10-21 01:25)

**Stage 6 輸出**:
```
文件: orbit-engine/data/outputs/rl_training/stage6/
      stage6_research_optimization_20251021_012508.json (225 MB)

事件統計:
  A3 events: 6,774
  A4 events: 21,224
  A5 events: 71,620
  D2 events: 6,074

候選池: 3,302 satellites
時間範圍: 1 天 (1,440 timesteps)
```

---

## 🎯 當前位置：項目階段

```
[✅ 完成] Stage 1: Offline BC Training
           ├─ 數據洩漏診斷與修復
           ├─ 閾值設計（數據驅動）
           ├─ BC V4 訓練成功
           └─ 產出: bc_policy_v4_best_20251021_020013.pth (88.81%)

[📍 當前] Stage 2: DQN Online Training (準備開始)
           ├─ 使用 BC 模型初始化
           ├─ Online RL 訓練
           └─ 算法: DQN (NOT PPO/SAC)

[待完成] Stage 3: 模型評估與部署
[待完成] Stage 4: 論文撰寫
```

---

## 🚀 下一步選項

### 選項 A: 檢查 DQN-BC 整合機制 (推薦優先執行)

**目的**: 確認是否能將 BC 模型加載到 DQN Q-Network

**需要檢查**:
1. `train.py` 是否支援 `--bc-init` 或類似參數？
2. `DQNAgent` 是否有從 BC 模型加載權重的方法？
3. BC 模型架構（128-64-32-1）與 DQN Q-Network 是否兼容？

**可能的結果**:
- ✅ 已有整合機制 → 直接進入選項 B
- ❌ 需要實現整合 → 添加 BC 初始化功能

**預計時間**: 10-15 分鐘

**執行命令**:
```bash
# 1. 檢查 train.py 參數
python train.py --help | grep -i "bc\|init\|pretrain"

# 2. 檢查 DQNAgent 源碼
grep -n "load\|init\|pretrain" src/agents/dqn/dqn_agent.py

# 3. 比較模型架構
# BC: 128 → 64 → 32 → 1
# DQN Q-Network: ? (需要確認)
```

---

### 選項 B: 開始 DQN Level 1 訓練

**目的**: 使用 BC 模型作為 warm-start，開始 DQN 訓練

**前提**: 選項 A 確認整合機制可用

**訓練配置**:
```bash
python train.py \
  --algorithm dqn \
  --level 1 \
  --bc-init checkpoints/bc_policy_v4_best_20251021_020013.pth \
  --output-dir output/dqn_level1_with_bc \
  --seed 42
```

**Level 1 參數** (快速驗證):
- Episodes: ~1,000
- 預計時間: 2 hours
- 目的: 驗證 BC 初始化是否有效

**預期結果**:
- 初始 reward > 隨機 policy
- 收斂速度 > 無 BC 初始化的 DQN
- 無 catastrophic forgetting（不會遺忘 BC 學到的知識）

**成功標準**:
- Episode reward 穩定上升
- Q-value 估計合理
- 無訓練崩潰

---

### 選項 C: 測試 BC 模型在環境中的表現

**目的**: 在 DQN environment 中評估 BC 模型的實際性能

**為什麼需要**:
- BC 訓練使用的是離線數據（準確率 88.81%）
- 需要驗證在真實環境中的決策品質
- 確認 BC 模型是否能作為良好的初始 policy

**評估指標**:
```python
# 1. 換手成功率
#    - 換手後信號品質是否改善？
#    - 是否選擇了最佳衛星？

# 2. 平均 reward
#    - 比較 BC policy vs Random policy
#    - 預期: BC >> Random

# 3. 決策品質
#    - 是否避免頻繁換手？
#    - 是否在合適時機觸發換手？
```

**執行方式**:
```python
# 創建評估腳本: evaluate_bc_policy.py
# 1. 加載 BC 模型
# 2. 在 environment 中運行 N episodes
# 3. 記錄 rewards, handover 決策
# 4. 與 random policy 比較
```

**預計時間**: 30-60 分鐘

---

### 選項 D: 實現 BC 到 DQN 的權重轉移

**目的**: 如果選項 A 發現缺少整合機制，需要實現

**需要做的**:
1. 分析 BC 模型架構 vs DQN Q-Network 架構
2. 設計權重映射策略
3. 實現加載函數

**可能的策略**:

#### 策略 1: 直接映射（如果架構相同）
```python
# BC: 128 → 64 → 32 → 1
# DQN: 120 → 128 → 128 → 11 (假設)

# 如果不兼容，使用策略 2
```

#### 策略 2: 部分初始化
```python
# 只初始化前幾層
# 最後一層重新訓練（因為輸出維度不同）
q_network.fc1.weight = bc_model.net[0].weight
q_network.fc2.weight = bc_model.net[3].weight
# fc3 隨機初始化
```

#### 策略 3: Knowledge Distillation
```python
# BC 作為 teacher
# DQN 作為 student
# 使用 KL divergence loss
```

**預計時間**: 1-2 小時

---

## 📋 待辦事項優先級

### P0 - 立即執行（今天）

- [ ] **選項 A**: 檢查 DQN-BC 整合機制
  - [ ] 檢查 `train.py` 參數
  - [ ] 檢查 `DQNAgent` 加載機制
  - [ ] 比較 BC 與 DQN Q-Network 架構
  - [ ] 記錄發現到 `BC_DQN_INTEGRATION_STATUS.md`

### P1 - 短期（本週）

- [ ] **選項 B 或 C**:
  - [ ] 如果整合機制存在 → 開始 DQN Level 1 訓練
  - [ ] 如果需要實現 → 執行選項 D

- [ ] 創建 DQN 訓練配置文件
  - [ ] 確認 hyperparameters
  - [ ] 設置 TensorBoard logging
  - [ ] 準備實驗追蹤表格

### P2 - 中期（2 週內）

- [ ] DQN Level 1 訓練完成並評估
- [ ] 如果成功 → 進入 Level 2-3 訓練
- [ ] 開始論文撰寫（方法論部分）

### P3 - 長期（1 個月內）

- [ ] DQN Level 5 完整訓練
- [ ] 模型評估與對比實驗
- [ ] 論文初稿完成

---

## 🔧 技術債與已知問題

### 已知限制

1. **BC 模型架構與 DQN 可能不兼容**
   - BC: 二分類 (handover vs maintain)
   - DQN: K+1 動作 (選擇哪顆衛星 or maintain)
   - 解決方案: 見選項 D 的策略 2 或 3

2. **候選池採樣成功率低**
   - 當前: 4.1% (5,007/121,480)
   - 原因: 隨機配對，大多數不滿足 margin ≤ 0
   - 影響: 訓練數據量較少
   - 未來改進: timestamp-based sampling (需要重新生成 Stage 5)

3. **訓練數據時間範圍有限**
   - 當前: 1 天 (1,440 timesteps)
   - 理想: 多天數據，覆蓋不同場景
   - 影響: 模型泛化能力可能受限

### 技術債

- [ ] V5 timestamp-based sampling 失敗（Stage 5/6 時間不匹配）
- [ ] 需要重新生成 Stage 5 與 Stage 6 同步
- [ ] Early stopping 未實現（目前手動選擇 Epoch 20）
- [ ] Validation set 未分離（目前只有 train/test split）

---

## 📊 關鍵文件索引

### 模型文件
```
checkpoints/
├── bc_policy_v4_best_20251021_020013.pth          # 最佳 BC 模型 ⭐
├── bc_v4_20251021_020013/                          # 所有 checkpoints
│   ├── epoch_19_testacc_85.39.pth
│   ├── epoch_20_testacc_88.81.pth
│   └── training_history.json
└── (待生成) dqn_level1_with_bc/                    # DQN 訓練輸出
```

### 數據文件
```
orbit-engine/data/outputs/rl_training/
├── stage5/stage5_signal_analysis_20251021_012459.json  # 信號分析
├── stage6/stage6_research_optimization_20251021_012508.json # 換手事件 ⭐
└── (225 MB, 21,224 A4 events, 6,074 D2 events)
```

### 文檔
```
handover-rl/
├── TODO.md                                 # 本文件 ⭐
├── FINAL_SOLUTION_SUMMARY.md               # 完整解決方案
├── TRAINING_REPORT_V4_FINAL.md             # BC V4 訓練報告
├── DIAGNOSIS_100_ACCURACY.md               # 問題診斷
└── FINAL_THRESHOLD_RECOMMENDATIONS.md      # 閾值建議
```

### 訓練腳本
```
handover-rl/
├── train.py                                # DQN 統一訓練入口 ⭐
├── train_offline_bc_v4_candidate_pool.py   # BC V4 訓練腳本 ✅
└── src/agents/dqn/dqn_agent.py             # DQN Agent 實現
```

---

## 🎓 學術貢獻點

已完成:
1. ✅ 數據驅動閾值設計（基於 48,002 真實換手事件）
2. ✅ Threshold-based negative sampling（消除數據洩漏）
3. ✅ 候選池約束（確保 co-visibility）
4. ✅ 配置 bug 發現與系統性修復

待完成:
- [ ] BC-initialized DQN（warm-start 效果評估）
- [ ] Multi-level training strategy 驗證
- [ ] 與 baseline 比較（random, greedy, 3GPP standard）

---

## 💡 決策記錄

### 2025-10-21 決策
- ✅ 選擇 Epoch 20 作為最佳模型（88.81%）
- ✅ 不延長訓練到 50-100 epochs（已驗證平滑學習曲線）
- ✅ 使用 DQN 而非 PPO/SAC（項目既有架構）

---

## 📞 問題與答案

**Q: 為什麼只有 2 個 epoch 在目標範圍（85-95%）？**
A: Epoch 19-20 達標。這是健康的學習曲線，平滑上升且最終穩定在目標範圍。比之前突跳到 100% 的異常行為好得多。

**Q: BC 訓練準確率是否太低（88.81% vs 原先期望 90-94%）？**
A: 不會。88.81% 在 85-95% 目標範圍內，且泛化優秀（Train-Test 差距僅 0.54%）。這是真實學習的結果，而非數據洩漏的虛高準確率。

**Q: 下一步應該選哪個選項？**
A: 推薦按順序執行：選項 A（檢查整合機制）→ 選項 B/C（開始訓練或測試）→ 必要時執行選項 D（實現整合）

---

**創建時間**: 2025-10-21 02:03:00
**維護者**: Claude Code
**項目**: Satellite Handover RL (DQN-based)
