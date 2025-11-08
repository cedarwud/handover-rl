# Quick Start Guide - 快速開始指南

**TL;DR**: 不要一開始就跑35小時！用漸進式驗證策略。

---

## 🚀 馬上開始

### 方法1: 使用快速訓練腳本（推薦）

```bash
# 快速驗證系統 (10分鐘)
./quick_train.sh 0

# 驗證訓練邏輯 (2小時)
./quick_train.sh 1

# 驗證效果 (10小時) - 推薦開始
./quick_train.sh 3

# 完整訓練 (35小時) - 論文最終實驗
./quick_train.sh 5
```

### 方法2: 手動指定參數

```bash
# Level 1: Quick Test (2小時)
python3 train_online_rl.py \
    --num-satellites 20 \
    --num-episodes 100 \
    --overlap 0.5 \
    --output-dir output/level1_quick_test \
    --log-interval 10
```

---

## 📊 訓練層級對比

| Level | 時間 | 衛星數 | Episodes | 適用場景 |
|-------|------|--------|---------|---------|
| **0** | 10分鐘 | 10 | 10 | 驗證系統運行 |
| **1** | 2小時 | 20 | 100 | 驗證訓練邏輯 ⭐ 先跑這個 |
| **2** | 6小時 | 50 | 300 | 調試超參數 |
| **3** | 10小時 | 101 | 500 | 驗證效果（論文初稿可用） |
| **4** | 21小時 | 101 | 1000 | 建立基準（論文實驗可用） |
| **5** | 35小時 | 101 | 1700 | 完整訓練（論文最終實驗） |

---

## 🎯 推薦工作流程

### Week 1: 快速迭代（總共 <20小時）

```bash
# Day 1 上午: 驗證系統 (10分鐘)
./quick_train.sh 0

# Day 1 下午: 看到學習曲線 (2小時)
./quick_train.sh 1

# Day 2: 調整超參數 (6小時)
./quick_train.sh 2

# Day 3: 驗證收斂 (10小時)
./quick_train.sh 3
```

**這4天你就能得到可用的初步結果！**

### Week 2-3: 完整實驗（可選）

```bash
# 如果需要更穩定的結果 (21小時)
./quick_train.sh 4

# 論文最終投稿前 (35小時)
./quick_train.sh 5
```

---

## 🔧 可調整的參數

### 主要參數

```bash
--num-satellites  # 衛星數量（預設: 101 Starlink）
--num-episodes    # Episode 數量（直接影響訓練時間）
--overlap         # Overlap 比例（預設: 0.5 = 50%）
--output-dir      # 輸出目錄
--log-interval    # 日誌間隔
```

### 參數對訓練時間的影響

1. **Episodes** (最關鍵)
   - 100 episodes → ~2小時
   - 500 episodes → ~10小時
   - 1700 episodes → ~35小時

2. **衛星數** (次要)
   - 10顆 → 略快（但多樣性低）
   - 20顆 → 適中（快速驗證）
   - 50顆 → 適中（開發調試）
   - 101顆 → 標準（完整訓練）

3. **Overlap** (影響時間覆蓋，不影響訓練時間)
   - 0.0 (無overlap) → 覆蓋更長時間
   - 0.5 (50% overlap) → 數據更密集

---

## 📁 輸出結構

訓練完成後會產生：

```
output/
└── level1_quick_test/          # 你指定的輸出目錄
    ├── checkpoints/            # 模型檢查點
    │   ├── checkpoint_ep100.pth
    │   ├── best_model.pth
    │   └── final_model.pth
    ├── logs/                   # TensorBoard 日誌
    │   └── events.out.tfevents.*
    └── training.log            # 訓練日誌
```

### 查看訓練進度

```bash
# 查看訓練日誌
tail -f output/level1_quick_test/logs/training.log

# 啟動 TensorBoard
tensorboard --logdir output/level1_quick_test/logs
# 然後在瀏覽器打開 http://localhost:6006
```

---

## 🐛 常見問題

### Q: 訓練時間太長怎麼辦？

**A**: 不要一開始就跑完整訓練！

- 開發時用 Level 1-2 (2-6小時)
- 驗證時用 Level 3 (10小時)
- 只在最後跑 Level 5 (35小時)

### Q: 衛星數量可以減少嗎？

**A**: 可以！

- Level 0-1: 10-20顆（快速測試）
- Level 2: 50顆（開發調試）
- Level 3-5: 101顆（完整訓練）

### Q: 為什麼有些 episode reward 是 -1.00？

**A**: 正常現象

- 衛星數少時可見度低
- 某些時間沒有衛星經過
- 隨著訓練進行會改善

### Q: Overlap 設多少比較好？

**A**: 看使用場景

- **快速測試**: 0.0 (無overlap) - 覆蓋更多時間
- **標準訓練**: 0.5 (50% overlap) - 符合 config 設計
- **數據密集**: 0.7-0.8 - 更多數據點

### Q: 時間覆蓋範圍夠嗎？

**A**: 夠了！

- Level 1 (100 episodes): 6.6天 - 測試用
- Level 3 (500 episodes): 16.5天 - 驗證用
- Level 4 (1000 episodes): 33天 - 論文可用
- Level 5 (1700 episodes): 56天 - 最佳

不需要涵蓋完整的288天軌道重複週期！

---

## 💡 最佳實踐

### 1. 先跑 Level 1 看看

```bash
./quick_train.sh 1  # 2小時
```

**檢查**:
- ✅ 代碼無錯誤
- ✅ Reward 有上升趨勢
- ✅ Loss 曲線合理

### 2. 調整參數後跑 Level 2

```bash
# 修改超參數後
./quick_train.sh 2  # 6小時
```

**對比**:
- 不同學習率
- 不同獎勵權重
- 不同網絡架構

### 3. 驗證效果用 Level 3

```bash
./quick_train.sh 3  # 10小時
```

**評估**:
- 與3GPP A3 baseline 對比
- 論文初稿實驗
- 確認訓練收斂

### 4. 最後跑 Level 4-5

```bash
# 論文最終實驗
./quick_train.sh 5  # 35小時
```

---

## 📈 預期效果

### Level 1 (100 episodes, 2小時)
- Reward: 可能開始上升
- 收斂: ❌ 未收斂
- 論文可用: ❌

### Level 3 (500 episodes, 10小時)
- Reward: 明顯提升
- 收斂: ✅ 基本收斂
- 論文可用: ⚠️ 初稿可用

### Level 5 (1700 episodes, 35小時)
- Reward: 最佳
- 收斂: ✅ 完全收斂
- 論文可用: ✅ 最終實驗

---

## 🎓 完整指令範例

### 自訂訓練

```bash
python3 train_online_rl.py \
    --num-satellites 50 \      # 使用50顆衛星
    --num-episodes 300 \       # 訓練300個episodes
    --overlap 0.5 \            # 50% overlap
    --seed 42 \                # 隨機種子
    --output-dir output/my_training \
    --log-interval 20 \        # 每20個episodes記錄一次
    --checkpoint-interval 100  # 每100個episodes存檔
```

### 背景執行

```bash
nohup ./quick_train.sh 3 > training.log 2>&1 &

# 查看進度
tail -f training.log
```

---

## 📚 更多資訊

- **詳細訓練層級**: 見 `TRAINING_LEVELS.md`
- **星座選擇說明**: 見 `CONSTELLATION_CHOICE.md`
- **數據依賴關係**: 見 `DATA_DEPENDENCIES.md`

---

**重點提醒**:

✅ **不要一開始就跑35小時！**

從 Level 1 (2小時) 開始，逐步驗證，最後才跑完整訓練。

---

**Date**: 2025-10-19
**Status**: ✅ Ready to use
