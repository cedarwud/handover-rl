# 訓練腳本目錄

本目錄包含特定的訓練方法腳本。

---

## 📁 目錄結構

```
training/
├── online_rl/         # 在線強化學習訓練
└── bc/                # Behavior Cloning 訓練
```

---

## 🔧 腳本說明

### online_rl/train_online_rl.py
**Phase 3: 在線 RL 訓練**

使用 DQN agent 在多衛星環境中進行在線學習。

**特點**:
- 在線 RL（agent 探索環境）
- 無預標記數據
- 真實 TLE 數據 + 完整物理模型
- 可重複實驗（seed 控制）

**使用**:
```bash
cd /home/sat/satellite/handover-rl
python scripts/training/online_rl/train_online_rl.py [options]
```

**注意**: 這是特定階段的訓練腳本。對於統一的訓練入口，請使用根目錄的 `train.py`。

---

### bc/train_offline_bc_v4_candidate_pool.py
**Offline Behavior Cloning V4 - Candidate Pool Based**

使用候選池方法的離線行為克隆訓練。

**特點**:
- Positive samples: Stage 6 A4/D2 events (margin > 0, 已觸發)
- Negative samples: 從候選池隨機採樣，計算真實 trigger margin < 0
- 目標準確率: 85-95%（消除數據洩漏）

**使用**:
```bash
cd /home/sat/satellite/handover-rl
python scripts/training/bc/train_offline_bc_v4_candidate_pool.py [options]
```

**狀態**: V4 是成功版本（達到 88.81% 準確率）

**其他版本**: V1, V2, V3, V5 已歸檔在 `archive/scripts/`

---

## 🎯 主訓練腳本

**推薦使用根目錄的統一訓練腳本**:

```bash
# DQN 訓練（Level 1-5）
python train.py --algorithm dqn --level 5 --output-dir output/level5

# 評估模型
python evaluate.py --model checkpoints/best_model.pth --algorithm dqn
```

詳見: 根目錄 `train.py` 和 `evaluate.py`

---

## 📚 相關文檔

- BC 訓練報告: `docs/reports/FINAL_SOLUTION_SUMMARY.md`
- 訓練級別說明: `docs/training/TRAINING_LEVELS.md`
- 專案狀態: `docs/PROJECT_STATUS.md`

---

**最後更新**: 2025-11-08
