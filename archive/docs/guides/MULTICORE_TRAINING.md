# 多核心訓練使用指南

## 概述

現在訓練支持多核心並行化！使用 `--num-envs` 參數來指定並行環境數量。

## 快速開始

### 基本用法

```bash
# 單核心（原始行為）
python train.py --algorithm dqn --level 4 \
  --config config/epsilon_fixed_config.yaml \
  --output-dir output/single_core \
  --seed 42

# 8 核心（推薦）
python train.py --algorithm dqn --level 4 \
  --config config/epsilon_fixed_config.yaml \
  --output-dir output/8_cores \
  --num-envs 8 \
  --seed 42

# 30 核心（最大速度）
python train.py --algorithm dqn --level 4 \
  --config config/epsilon_fixed_config.yaml \
  --output-dir output/30_cores \
  --num-envs 30 \
  --seed 42
```

## 性能預期

| 配置 | 加速比 | 1000 episodes | 10000 episodes |
|------|--------|---------------|----------------|
| 1 核心 | 1.0x | 3.9 小時 | 39 小時 |
| 8 核心 | ~3.0x | 1.3 小時 | 13 小時 |
| 16 核心 | ~3.5x | 1.1 小時 | 11 小時 |
| 30 核心 | ~3.8x | 1.0 小時 | 10 小時 |

**注意**: 由於 Amdahl's Law，加速比有上限（約 5x），因為神經網路更新是序列的。

## 推薦配置

### 開發測試
- 使用 **4-8 核心**
- 平衡速度和資源利用
- CPU 效率約 40-50%

### 正式實驗
- 使用 **8-16 核心**
- 最佳性價比
- 顯著加速且穩定

### 最大速度
- 使用 **30 核心**
- 用滿所有可用 CPU
- 適合趕時間的實驗

## 記憶體需求

| 核心數 | 記憶體需求 |
|--------|-----------|
| 1 | ~0.3 GB |
| 8 | ~1.4 GB |
| 16 | ~2.5 GB |
| 30 | ~4.7 GB |

您的系統有 31 GB RAM，完全足夠。

## 測試腳本

運行測試來驗證安裝：

```bash
./test_multicore.sh
```

這會測試 1、8、30 核心的配置，並比較性能。

## 技術細節

### 實現方式

使用 Gymnasium 的 `AsyncVectorEnv` 來並行運行多個環境實例：

1. **環境並行化**: 每個核心運行一個獨立的環境實例
2. **經驗收集**: 從所有環境並行收集經驗
3. **神經網路更新**: 序列執行（Amdahl 瓶頸）
4. **自動批量處理**: Trainer 自動處理向量化數據

### 與單核心的差異

- **單核心**: 每個 episode 序列執行
- **多核心**: 同時運行多個 episode，收集更多經驗

### 訓練結果

多核心訓練的結果應該與單核心**幾乎相同**：
- 相同的 hyperparameters
- 相同的隨機種子
- 相同的收斂行為

唯一差異：
- 多核心收集經驗更快
- 經驗可能略有不同順序（但整體分布相同）

## 故障排除

### OOM (記憶體不足)

如果遇到記憶體錯誤：

```bash
# 降低核心數
--num-envs 16  # 改為 16
--num-envs 8   # 或改為 8
```

### 訓練變慢

如果多核心反而更慢（不太可能）：

1. 檢查 CPU 使用率：`htop`
2. 降低核心數
3. 檢查磁碟 I/O

### 不穩定/崩潰

如果訓練不穩定：

1. 測試較少核心數（4 或 8）
2. 檢查日誌文件
3. 報告 issue

## 當前訓練不受影響

修改只是添加新功能，不會影響：
- 正在運行的訓練 (PID 297058)
- 現有的單核心訓練腳本
- 之前訓練的模型

預設 `--num-envs 1` 保持原有行為。

## 下一步

1. 等待當前訓練完成（07:55）
2. 運行測試腳本驗證多核心功能
3. 用 30 核心重新訓練 epsilon fix

## 參考

- Gymnasium AsyncVectorEnv: https://gymnasium.farama.org/api/vector/
- Amdahl's Law: https://en.wikipedia.org/wiki/Amdahl%27s_law
- Stable Baselines3 Vectorized Envs: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
