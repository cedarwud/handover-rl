# 訓練結果可視化指南

## 📊 典型的強化學習訓練可視化

### 1. Episode Reward 曲線 ⭐ 最重要

```
Reward
  10 |                                    ___---‾‾‾
   5 |                        ___---‾‾‾‾‾
   0 |        ___---‾‾‾‾‾----
  -5 |  _---‾‾
 -10 |--                                     | Episode 920
     |________________________________________|
     0     500    1000   1500   1700
                Episodes
```

**用途**：
- 展示訓練進展（reward 是否提升）
- 判斷模型是否在學習
- 論文中的核心圖表

**預期結果**：
- 開始：reward ≈ -10（隨機策略）
- 訓練中：逐漸上升
- 收斂：reward ≈ +5 到 +10（取決於環境）

---

### 2. Training Loss 曲線 ⭐ Episode 920 驗證

```
Loss (log scale)
10^6 |    舊版本
     |              X  ← Episode 920 爆炸！
     |             /|
  10 |------------/
   1 |___________/
     |_____________________________
     |  新版本（修復後）
  10 |       
   1 |_____～～～～～～～～～～～～  ← 穩定！
     |________________________________________|
     0     500    920   1500   1700
```

**用途**：
- **關鍵：驗證 Episode 920 bug 是否修復**
- 舊版：Episode 920 處 loss 爆炸到 1e6+
- 新版：loss < 10，穩定

**預期結果**：
- 開始：loss ≈ 100（隨機初始化）
- 訓練中：逐漸下降到 0.5-2.0
- Episode 920：**loss < 10** ✅（vs 舊版 > 1e6 ❌）

---

### 3. Episode 920 局部放大圖 ⭐ 論文關鍵

```
Loss
 10 |
    |         🔴 關鍵驗證區
  5 |     ┌─────────┐
    |     │   920   │
  1 |_____│_________│_____
    |_____|_________|_____|
      900   920   940
         Episodes

舊版本：Episode 920 loss = 1,234,567 ❌
新版本：Episode 920 loss = 0.89       ✅
```

**用途**：
- 論文中展示 bug 修復效果
- 對比修復前後
- 證明數值穩定性改進

---

### 4. Handover 次數趨勢

```
Handovers
 15 |●  
    |  ●    ●
 10 |   ● ●   ● 
    |      ●   ●  ●
  5 |           ●  ●  ●___
    |                    ‾‾‾
  0 |___________________________
    0      500     1000    1700
           Episodes
```

**用途**：
- 評估策略效率
- handover 次數越少 = 策略越穩定
- 論文中的性能指標

**預期結果**：
- 開始：15-20 次/episode（頻繁切換）
- 訓練後：5-8 次/episode（合理策略）

---

### 5. 性能對比表格（配合圖表）

```
┌──────────────┬─────────┬─────────┬──────────┐
│   Method     │ Reward  │  Loss   │ Handover │
├──────────────┼─────────┼─────────┼──────────┤
│ 舊版本       │  N/A    │ CRASHED │   N/A    │
│ (Episode 920)│         │ @920    │          │
├──────────────┼─────────┼─────────┼──────────┤
│ 新版本       │  +7.2   │  0.89   │   6.3    │
│ (修復後)     │  ±2.1   │  ±0.3   │   ±1.2   │
├──────────────┼─────────┼─────────┼──────────┤
│ Baseline     │  +1.5   │  1.2    │   8.1    │
│ (貪心策略)   │  ±1.8   │  ±0.4   │   ±1.5   │
└──────────────┴─────────┴─────────┴──────────┘
```

---

### 6. 訓練量對比圖（論文學術正當性）

```
Training Steps
2.0M |                     
     | 理論完整軌道      ██████
1.0M |           MuJoCo標準  ══════
     |                    
0.5M | 我們的方案        ████  ✅ 可接受
     |                    
0.1M | 原始版本(bug)     ██  ❌ 不足
     |_________________________________
        舊版   新版  標準  完整
```

**用途**：
- 證明訓練量充足
- 論文中說明為何 408K steps 是合理的

---

## 🎯 針對本項目的關鍵可視化

### 必須有的圖表（論文）

1. **Episode Reward 學習曲線**
   - 證明模型在學習
   - 展示收斂性能

2. **Episode 920 Loss 對比** ⭐ 最重要
   - 舊版 vs 新版
   - 證明 bug 修復
   - 數值穩定性改進

3. **訓練量說明圖**
   - 408K steps vs 標準
   - 學術正當性

### 可選的圖表（補充）

4. **Handover 次數分析**
   - 策略效率
   - 與 baseline 對比

5. **RSRP/信號質量趨勢**
   - 領域特定指標
   - 證明物理模型有效

6. **計算時間分析**
   - 多核加速效果
   - 成本分析

---

## 💡 如何使用這些可視化

### 在論文中

**實驗結果章節**：
```
Figure 1: Training curves showing (a) episode reward, 
          (b) training loss, (c) handover frequency.

Figure 2: Episode 920 bug verification. Comparison of 
          loss values before and after numerical stability 
          fixes. Old version crashes at episode 920 
          (loss > 1e6), while the fixed version maintains 
          stable loss (< 1).

Figure 3: Training scale justification. Our approach 
          achieves 408K training steps using 20-minute 
          episodes representing realistic user sessions.
```

**故事線**：
1. 問題：Episode 920 訓練崩潰
2. 診斷：數值不穩定導致 loss 爆炸
3. 解決：4層防護 + episode 調整
4. 驗證：**圖表證明修復成功**
5. 結果：性能超越 baseline

### 在演講/報告中

**投影片結構**：
```
Slide 1: 問題發現
  → Episode 920 crash 截圖/圖表

Slide 2: 診斷過程
  → Loss 爆炸曲線圖

Slide 3: 解決方案
  → 4層防護架構圖

Slide 4: 驗證結果 ⭐
  → Episode 920 前後對比圖
  → Loss < 1 vs > 1e6

Slide 5: 最終性能
  → Reward 曲線
  → 與 baseline 對比
```

---

## 🔧 如何生成這些圖表

### 方法 1: Python + Matplotlib（推薦）

```python
import matplotlib.pyplot as plt
import numpy as np

# 從 log 提取數據
rewards = extract_rewards('training_level5_20min_final.log')
losses = extract_losses('training_level5_20min_final.log')

# 繪製 Episode 920 驗證圖
plt.figure(figsize=(10, 6))
plt.plot(episodes, losses, label='Training loss')
plt.axvline(x=920, color='red', linestyle='--', label='Episode 920')
plt.axhline(y=10, color='green', linestyle=':', label='Safe threshold')
plt.xlabel('Episode')
plt.ylabel('Loss (log scale)')
plt.yscale('log')
plt.legend()
plt.title('Episode 920 Bug Verification')
plt.savefig('episode920_verification.png')
```

### 方法 2: TensorBoard（如果已集成）

```bash
tensorboard --logdir=output/level5_20min_final/logs
```

瀏覽器打開：http://localhost:6006

### 方法 3: 手動分析（備選）

```bash
# 提取 reward 數據
grep "Episode.*reward=" training_level5_20min_final.log | \
  awk '{print $2, $4}' > rewards.txt

# 用 Excel/Google Sheets 繪圖
```

---

## 📋 可視化清單（訓練完成後）

- [ ] Episode Reward 曲線（PNG, 300 DPI）
- [ ] Training Loss 曲線（PNG, 300 DPI）
- [ ] Episode 920 局部放大圖（PNG, 300 DPI）
- [ ] 舊版 vs 新版對比圖（PNG, 300 DPI）
- [ ] Handover 次數趨勢（可選）
- [ ] 訓練量說明圖（可選）
- [ ] 性能對比表格（LaTeX/Markdown）

---

## 🎨 圖表風格建議

### 學術論文標準

- **解析度**: 300 DPI（印刷品質）
- **格式**: PDF 向量圖 或 PNG
- **字體**: 與論文一致（通常 Times New Roman, 10-12pt）
- **顏色**: 考慮黑白印刷（用線型區分）
- **標題**: 圖下方，完整說明

### 投影片

- **解析度**: 150 DPI（屏幕顯示）
- **格式**: PNG
- **字體**: 大而清晰（18-24pt）
- **顏色**: 高對比度
- **動畫**: 可以分步顯示

---

**總結**：可視化是**訓練結果呈現的關鍵工具**，特別是：
1. 證明訓練有效（reward 提升）
2. **證明 bug 修復（Episode 920 穩定）** ⭐
3. 學術正當性（訓練量說明）
4. 與 baseline 對比（性能優勢）

沒有好的可視化，論文/報告會缺乏說服力！
