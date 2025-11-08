# 🔍 Level 6 訓練監控指南

**訓練時間**: 4.3 天 (104 小時)
**總 Episodes**: 17,000
**關鍵檢查點**: Episode 920, 5000, 10000, 15000

---

## 🚨 自動監控（推薦）

### 方案 1: 使用監控腳本（最推薦）

**啟動監控**:
```bash
# 在另一個終端執行
./monitor_training.sh
```

**功能**:
- ✅ 每10分鐘自動檢查
- ✅ 檢測 NaN/Inf 錯誤
- ✅ 檢測 Large Loss 警告
- ✅ 顯示訓練進度
- ✅ 監控 GPU 記憶體
- ✅ 檢測訓練卡住
- ✅ 自動記錄警報到 `training_alerts.txt`

**停止監控**:
```bash
Ctrl+C
```

---

## 👀 手動檢查

### 快速檢查（隨時執行）

```bash
./quick_check.sh
```

**輸出內容**:
1. 當前進度
2. 錯誤統計
3. Episode 920 狀態
4. 最近10個 episodes
5. GPU 狀態
6. 訓練時間估算
7. 最新 checkpoint

**建議頻率**: 
- 前100 episodes: 每小時
- Episode 100-920: 每3小時
- Episode 920-950: **每小時**（關鍵區域）
- Episode 950+: 每6小時

---

## 📊 關鍵檢查點

### Checkpoint 1: Episode 500 (~3小時)

**檢查內容**:
```bash
./quick_check.sh

# 或手動查看
tail -50 long_training_17k.log
grep "Episode  500" long_training_17k.log
```

**預期結果**:
- ✅ Loss 在 0.5-5.0 之間
- ✅ 沒有 NaN/Inf 錯誤
- ✅ 訓練穩定進行

**異常處理**:
- ❌ Loss > 100: 停止訓練，檢查配置
- ❌ 有 NaN/Inf: 立即停止

---

### Checkpoint 2: Episode 920 (~5小時) - **最關鍵**

**檢查內容**:
```bash
# Episode 915-925 期間密集監控
watch -n 60 "./quick_check.sh"

# 查看 Episode 920
grep "Episode  920" long_training_17k.log
```

**預期結果**:
- ✅ Loss ~0.5-2.0（與 Level 5 一致）
- ✅ Reward 正常範圍
- ✅ 沒有錯誤

**異常處理**:
- ❌ Loss > 100: 數值爆炸，修復失敗
- ❌ Loss = NaN: 立即停止，回報問題

---

### Checkpoint 3: Episode 5000 (~30小時, 1.3天)

**檢查內容**:
```bash
./quick_check.sh

# 檢查訓練曲線
grep "Episode.*loss=" long_training_17k.log | tail -100
```

**預期結果**:
- ✅ Loss 逐漸下降或穩定
- ✅ Reward 有改善趨勢
- ✅ Epsilon 逐漸降低

---

### Checkpoint 4: Episode 10000 (~61小時, 2.5天)

**檢查內容**:
```bash
./quick_check.sh

# 檢查收斂情況
grep "Episode.*loss=" long_training_17k.log | awk '{print $NF}' | tail -500
```

**預期結果**:
- ✅ Loss 已經收斂
- ✅ Policy 穩定

---

### Checkpoint 5: Episode 15000 (~92小時, 3.8天)

**最終衝刺階段**:
```bash
./quick_check.sh

# 確認即將完成
tail -100 long_training_17k.log
```

---

## 🔔 警報條件

### 立即停止訓練 ❌

1. **NaN/Inf 錯誤出現**
   ```bash
   grep "NaN/Inf Detection" long_training_17k.log
   ```

2. **Loss 爆炸 (> 1000)**
   ```bash
   grep "Large Loss Warning" long_training_17k.log
   ```

3. **Episode 920 失敗**
   ```bash
   # 如果 Episode 920 loss > 100
   grep "Episode  920" long_training_17k.log
   ```

4. **訓練卡住 (30分鐘沒更新)**
   ```bash
   ls -lth long_training_17k.log
   ```

### 需要關注 ⚠️

1. **Loss 不降反升**
   - 連續100 episodes loss 上升

2. **Reward 持續負值**
   - 10000 episodes 後 reward 仍然很差

3. **GPU 記憶體異常**
   - GPU OOM 錯誤

---

## 📈 TensorBoard 監控（如果可用）

**啟動 TensorBoard**:
```bash
tensorboard --logdir output/long_training_17k/logs --port 6006
```

**瀏覽器訪問**:
```
http://localhost:6006
```

**監控指標**:
- Loss 曲線
- Reward 曲線
- Epsilon 曲線
- Q-values 分布

---

## 💾 Checkpoint 保存

**配置**:
- **頻率**: 每 500 episodes
- **位置**: `output/long_training_17k/checkpoints/`
- **文件**: `checkpoint_ep500.pth`, `checkpoint_ep1000.pth`, ...

**檢查 checkpoints**:
```bash
ls -lh output/long_training_17k/checkpoints/
```

**預期文件**:
- Episode 500: `checkpoint_ep500.pth`
- Episode 1000: `checkpoint_ep1000.pth`
- ...
- Best model: `best_model.pth`
- Final: `final_model.pth`

---

## 🔧 從 Checkpoint 恢復

**如果訓練中斷**:
```bash
# 查找最新 checkpoint
ls -lt output/long_training_17k/checkpoints/checkpoint_ep*.pth | head -1

# 從 checkpoint 恢復（如果 train.py 支持）
python train.py \
  --algorithm dqn \
  --level 6 \
  --config config/diagnostic_config.yaml \
  --output-dir output/long_training_17k \
  --resume output/long_training_17k/checkpoints/checkpoint_ep5000.pth \
  --seed 42
```

---

## 📝 監控清單

### 每天必做檢查

**早上** (Day 1-4):
```bash
./quick_check.sh
```

**檢查內容**:
- [ ] 訓練是否還在運行
- [ ] 進度是否正常
- [ ] 沒有錯誤
- [ ] GPU 正常
- [ ] 最新 checkpoint 存在

**晚上**:
- [ ] 檢查 `training_alerts.txt`
- [ ] 查看訓練曲線趨勢

---

## ⏰ 時間規劃

| 時間點 | Episode | 檢查頻率 | 重點 |
|--------|---------|----------|------|
| 0-3h | 0-500 | 每小時 | 確認啟動成功 |
| 3-6h | 500-920 | 每2小時 | 準備 Episode 920 |
| **5-6h** | **915-925** | **每小時** | **密集監控 Episode 920** |
| 6-30h | 920-5000 | 每6小時 | 穩定訓練 |
| 30-60h | 5000-10000 | 每12小時 | 確認收斂 |
| 60-92h | 10000-15000 | 每12小時 | 最終階段 |
| 92-104h | 15000-17000 | 每6小時 | 準備完成 |

---

## 🚀 啟動流程

### Step 1: 啟動訓練
```bash
source venv/bin/activate

python train.py \
  --algorithm dqn \
  --level 6 \
  --config config/diagnostic_config.yaml \
  --output-dir output/long_training_17k \
  --seed 42 \
  2>&1 | tee long_training_17k.log &

echo "Training PID: $!"
```

### Step 2: 啟動監控（新終端）
```bash
./monitor_training.sh &
echo "Monitor PID: $!"
```

### Step 3: 第一次檢查（5分鐘後）
```bash
./quick_check.sh
```

### Step 4: 設置定時檢查（可選）
```bash
# 使用 cron 每小時檢查
(crontab -l 2>/dev/null; echo "0 * * * * cd $(pwd) && ./quick_check.sh >> hourly_check.log 2>&1") | crontab -
```

---

## 📞 緊急聯絡

**如果訓練出錯**:
1. 立即執行 `./quick_check.sh`
2. 保存 `training_alerts.txt`
3. 保存最後100行日誌: `tail -100 long_training_17k.log > error_log.txt`
4. 檢查最新 checkpoint 是否存在

**重要文件**:
- 日誌: `long_training_17k.log`
- 警報: `training_alerts.txt`
- Checkpoints: `output/long_training_17k/checkpoints/`
- 配置: `config/diagnostic_config.yaml`

---

**準備好了嗎？開始訓練！** 🚀
