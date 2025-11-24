# Training Guide - 預計算系統訓練指南

**日期**: 2025-11-08
**版本**: 3.0 (With Precompute System)

---

## ⚠️ 重要提醒

### 訓練前必須完成的步驟

**1. 生成預計算表** (一次性，約 30 分鐘)

```bash
python scripts/generate_orbit_precompute.py \
  --start-time "2025-10-07 00:00:00" \
  --end-time "2025-10-14 00:00:00" \
  --output data/orbit_precompute_7days.h5 \
  --config configs/diagnostic_config.yaml
```

**2. 啟用預計算模式**

編輯 `configs/diagnostic_config.yaml`:
```yaml
precompute:
  enabled: true  # 改為 true
  table_path: "data/orbit_precompute_7days.h5"
```

**3. 確認啟用成功**

運行訓練時應該看到：
```
✅ Precompute mode enabled - Training will be ~100x faster!
   Table: data/orbit_precompute_7days.h5
   Time range: 2025-10-07T00:00:00 to 2025-10-14T00:00:00
```

如果看到：
```
✅ Real-time calculation mode
⚠️  Training will be slow. Consider generating precompute table...
```
表示**未啟用**預計算，訓練會非常慢！

---

## 🎯 多級訓練策略

系統提供 **7 個訓練級別** (Level 0-6)，從快速測試到完整訓練。

### 訓練級別總覽

| Level | 名稱 | Episodes | 用途 | 推薦 |
|-------|------|----------|------|------|
| **0** | Smoke Test | 10 | 系統驗證 | 首次運行 |
| **1** | Quick Validation | 50 | 快速驗證 | ⭐ 開發 |
| **2** | Development | 200 | 開發迭代 | 調參 |
| **3** | Validation | 500 | 驗證有效性 | 論文草稿 |
| **4** | Baseline | 1000 | 建立基線 | 實驗對比 |
| **5** | Full Training | 1700 | 完整訓練 | 論文實驗 |
| **6** | Long-term | 17000 | 長期訓練 | ⭐ 發表 |

---

## 🚀 推薦訓練流程

### 階段 1: 系統驗證 (Level 0)

**目的**: 確認代碼運行無誤

```bash
python train.py \
  --algorithm dqn \
  --level 0 \
  --output-dir output/smoke_test \
  --config configs/diagnostic_config.yaml
```

**預期時間** (with precompute):
- **約 1-2 分鐘** (10 episodes)
- 實時模式: ~10 分鐘

**檢查項目**:
- ✅ 預計算表加載成功
- ✅ 環境正常運行
- ✅ Agent 可以訓練
- ✅ Checkpoint 正常保存

---

### 階段 2: 快速驗證 (Level 1) ⭐ 推薦

**目的**: 驗證訓練邏輯，觀察學習曲線

```bash
python train.py \
  --algorithm dqn \
  --level 1 \
  --output-dir output/level1_quick \
  --config configs/diagnostic_config.yaml
```

**預期時間** (with precompute):
- **約 5-10 分鐘** (50 episodes)
- 實時模式: ~8 小時

**適用場景**:
- ✅ 測試新的 hyperparameter
- ✅ 比較不同算法
- ✅ 快速迭代想法
- ✅ Debug reward function

**關鍵指標**:
- Episode reward 趨勢
- Handover count 變化
- Loss 是否收斂
- 是否有 NaN/Inf

---

### 階段 3: 開發迭代 (Level 2)

**目的**: 調整 hyperparameters 和 reward function

```bash
python train.py \
  --algorithm dqn \
  --level 2 \
  --output-dir output/level2_dev \
  --config configs/diagnostic_config.yaml
```

**預期時間** (with precompute):
- **約 20-40 分鐘** (200 episodes)
- 實時模式: ~33 小時

---

### 階段 4: 驗證有效性 (Level 3)

**目的**: 驗證方法有效性，準備論文草稿

```bash
python train.py \
  --algorithm dqn \
  --level 3 \
  --output-dir output/level3_validation \
  --config configs/diagnostic_config.yaml
```

**預期時間** (with precompute):
- **約 50 分鐘 - 1.5 小時** (500 episodes)
- 實時模式: ~83 小時

**關鍵檢查**:
- 與 baseline 比較
- Reward 提升百分比
- Convergence 分析

---

### 階段 5: 建立基線 (Level 4)

**目的**: 建立穩定基線供實驗比較

```bash
python train.py \
  --algorithm dqn \
  --level 4 \
  --output-dir output/level4_baseline \
  --config configs/diagnostic_config.yaml
```

**預期時間** (with precompute):
- **約 1.5-3 小時** (1000 episodes)
- 實時模式: ~167 小時 (7 天)

---

### 階段 6: 完整訓練 (Level 5)

**目的**: 論文實驗，publication-quality 結果

```bash
python train.py \
  --algorithm dqn \
  --level 5 \
  --output-dir output/level5_full \
  --config configs/diagnostic_config.yaml
```

**預期時間** (with precompute):
- **約 3-5 小時** (1700 episodes)
- 實時模式: ~283 小時 (12 天)

**論文使用**:
- 生成 learning curves
- 與多個 baselines 比較
- Ablation studies
- Statistical significance tests

---

### 階段 7: 長期訓練 (Level 6) ⭐ 發表推薦

**目的**: 達到 1M training steps (學術標準)

```bash
python train.py \
  --algorithm dqn \
  --level 6 \
  --output-dir output/level6_longterm \
  --config configs/diagnostic_config.yaml
```

**預期時間** (with precompute):
- **約 28-34 小時** (17000 episodes)
- 實時模式: ~2833 小時 (118 天！)

**學術意義**:
- 符合 MuJoCo benchmark 標準 (1M steps)
- Peer review 要求的充分訓練
- 確保完全收斂

---

## 📊 時間對比 (實時 vs 預計算)

### 未啟用預計算 (舊系統)

| Level | Episodes | 預估時間 | 實際 |
|-------|----------|----------|------|
| 0 | 10 | ~10 min | ❌ 太慢 |
| 1 | 50 | ~8 hours | ❌ 太慢 |
| 2 | 200 | ~33 hours | ❌ 太慢 |
| 3 | 500 | ~83 hours (3.5天) | ❌ 太慢 |
| 4 | 1000 | ~167 hours (7天) | ❌ 太慢 |
| 5 | 1700 | ~283 hours (12天) | ❌ 太慢 |
| 6 | 17000 | ~2833 hours (118天！) | ❌ 不可行 |

### 啟用預計算 (新系統) ✅

| Level | Episodes | 預估時間 | 加速比 |
|-------|----------|----------|--------|
| 0 | 10 | **~1-2 min** | **100x** |
| 1 | 50 | **~5-10 min** | **100x** |
| 2 | 200 | **~20-40 min** | **100x** |
| 3 | 500 | **~50 min - 1.5h** | **100x** |
| 4 | 1000 | **~1.5-3 hours** | **100x** |
| 5 | 1700 | **~3-5 hours** | **100x** |
| 6 | 17000 | **~28-34 hours** | **100x** |

**結論**:
- Level 6 從 **118 天 → 34 小時** ✅ 可行！
- 快速迭代成為可能

---

## 💡 建議的訓練順序

### 第一次訓練

```bash
# Day 1: 系統驗證和快速測試
# 1. 生成預計算表 (30 min)
python scripts/generate_orbit_precompute.py \
  --start-time "2025-10-07 00:00:00" \
  --end-time "2025-10-14 00:00:00" \
  --output data/orbit_precompute_7days.h5 \
  --config configs/diagnostic_config.yaml

# 2. 啟用預計算（編輯 config）
# 設置 precompute.enabled = true

# 3. Level 0: 煙霧測試 (1-2 min)
python train.py --algorithm dqn --level 0 --output-dir output/smoke_test

# 4. Level 1: 快速驗證 (5-10 min)
python train.py --algorithm dqn --level 1 --output-dir output/quick_val

# 5. 檢查結果
python evaluate.py \
  --model output/quick_val/checkpoints/best_model.pth \
  --algorithm dqn \
  --episodes 20
```

### 開發階段

```bash
# Level 2: 調整 hyperparameters (20-40 min each)
python train.py --algorithm dqn --level 2 --output-dir output/lr_2e5
# 修改 config，調整 learning_rate
python train.py --algorithm dqn --level 2 --output-dir output/lr_1e5
# 比較結果，選擇最佳配置
```

### 論文準備

```bash
# Level 3: 初步驗證 (50 min - 1.5h)
python train.py --algorithm dqn --level 3 --output-dir output/paper_draft

# Level 5: 完整實驗 (3-5 hours)
python train.py --algorithm dqn --level 5 --output-dir output/paper_exp1

# 比較 baselines
python evaluate.py --model output/paper_exp1/checkpoints/best_model.pth ...
```

### 論文提交前

```bash
# Level 6: 最終訓練 (28-34 hours)
# 建議：跑過夜 + 隔天
python train.py --algorithm dqn --level 6 --output-dir output/final_publication

# 完整評估
python evaluate.py \
  --model output/final_publication/checkpoints/best_model.pth \
  --algorithm dqn \
  --episodes 100 \
  --output-dir evaluation/final
```

---

## 🔍 監控訓練進度

### TensorBoard (推薦)

```bash
# 啟動 TensorBoard
tensorboard --logdir output/

# 瀏覽器打開
http://localhost:6006
```

### 日誌查看

```bash
# 實時查看訓練日誌
tail -f output/level1_quick/training.log

# 查看最新狀態
python tools/check_progress.sh
```

### 關鍵指標

監控以下指標：
- **Episode Reward**: 應該上升
- **Episode Length**: 應該趨於穩定
- **Handover Count**: 應該減少（避免 ping-pong）
- **Average RSRP**: 應該提升
- **Loss**: 應該收斂
- **Epsilon**: 應該遞減

---

## ⚠️ 常見問題

### Q1: 訓練時間比預期長？

**檢查**:
```python
# 查看日誌中是否有
✅ Precompute mode enabled - Training will be ~100x faster!
```

如果看到：
```
✅ Real-time calculation mode
```
表示**未啟用預計算**！

**解決**:
1. 檢查 `configs/diagnostic_config.yaml` 中 `precompute.enabled = true`
2. 檢查 `precompute.table_path` 是否正確
3. 確認 HDF5 文件存在：`ls -lh data/orbit_precompute_7days.h5`

### Q2: 出現 "Timestamp out of range" 錯誤？

**原因**: Episode 起始時間超出預計算表範圍

**解決**:
1. 生成更大的預計算表（例如 14 天）
2. 或調整 episode 起始時間範圍

### Q3: 訓練卡住不動？

**檢查**:
1. GPU 使用率：`nvidia-smi`
2. CPU 使用率：`htop`
3. 日誌中是否有錯誤

### Q4: Loss 爆炸或 NaN？

**已內建保護**:
- `enable_nan_check: true` (自動檢測)
- `q_value_clip: 100.0` (防止爆炸)
- Huber loss (更穩定)

如果仍出現問題：
1. 降低 learning rate
2. 增加 gradient clipping

---

## 📝 訓練檢查清單

### 開始訓練前

- [ ] 生成預計算表
- [ ] 啟用預計算模式（`config` 中設置）
- [ ] 確認 GPU 可用（如有）
- [ ] 確認磁盤空間足夠（checkpoints + logs）

### 訓練中

- [ ] 監控 TensorBoard
- [ ] 檢查 reward 趨勢
- [ ] 檢查 loss 收斂
- [ ] 定期保存 checkpoints

### 訓練後

- [ ] 評估最佳模型
- [ ] 與 baseline 比較
- [ ] 生成圖表（learning curves）
- [ ] 保存結果到 evaluation/

---

## 🎯 下一步

訓練完成後：

1. **評估模型**
   ```bash
   python evaluate.py \
     --model output/level5_full/checkpoints/best_model.pth \
     --algorithm dqn \
     --episodes 50
   ```

2. **生成論文圖表**
   ```bash
   python scripts/generate_paper_figures.sh
   ```

3. **比較不同方法**
   - DQN vs DDQN
   - DQN vs Baselines
   - Ablation studies

---

**準備好了嗎？開始訓練！** 🚀

建議從 **Level 0 (Smoke Test)** 開始，確認一切正常後再進行更長的訓練。
