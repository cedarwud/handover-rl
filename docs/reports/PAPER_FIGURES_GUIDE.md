# 論文圖表生成指南

本指南說明如何使用我們的圖表生成系統為論文創建高品質、符合學術出版標準的圖表。

## 📋 目錄

1. [快速開始](#快速開始)
2. [圖表系統概覽](#圖表系統概覽)
3. [詳細使用說明](#詳細使用說明)
4. [論文中的圖表布局建議](#論文中的圖表布局建議)
5. [常見問題](#常見問題)

---

## 🚀 快速開始

### 一鍵生成所有圖表

訓練完成後，運行以下命令：

```bash
./generate_paper_figures.sh
```

這將自動生成：
- ✅ Episode 920 對比圖（核心技術貢獻）
- ✅ Learning Curves（標準 RL 圖表）
- ✅ Handover 分析圖（領域特定）
- ✅ 性能對比表格（LaTeX + Markdown）

所有圖表將儲存在 `figures/` 目錄，表格儲存在 `tables/` 目錄。

### 快速模式（僅生成最重要圖表）

```bash
./generate_paper_figures.sh --quick
```

僅生成：
- Episode 920 對比圖
- Learning Curve
- 多指標曲線

---

## 📊 圖表系統概覽

我們的圖表系統包含以下組件：

### 1. 數據提取工具

**腳本**: `scripts/extract_training_data.py`

從訓練日誌中提取結構化數據。

```bash
python scripts/extract_training_data.py training_level5_20min_final.log \
    --output data/training_metrics.csv \
    --stats
```

**輸出**: CSV 檔案，包含 episode、reward、loss、handovers 等數據

---

### 2. 論文樣式配置

**腳本**: `scripts/paper_style.py`

提供論文級的圖表樣式配置，符合 IEEE / NeurIPS / ICML 標準。

**特點**:
- 色盲友好配色
- 300 DPI 印刷品質
- PDF vector 格式輸出
- 符合學術出版字型和尺寸規範

**可用樣式**:
- `'default'`: 通用學術樣式（推薦）
- `'ieee'`: IEEE 期刊/會議樣式
- `'neurips'`: NeurIPS/ICML/ICLR 樣式
- `'nature'`: Nature 期刊樣式

---

### 3. Episode 920 對比圖（核心圖表） ⭐

**腳本**: `scripts/plot_episode920_comparison.py`

**這是論文中最重要的圖表**，用於證明您的數值穩定性修復有效。

**用法**:

```bash
# 單獨生成（僅新版本）
python scripts/plot_episode920_comparison.py \
    --new training_level5_20min_final.log \
    --output figures/episode920_comparison

# 對比舊版本和新版本
python scripts/plot_episode920_comparison.py \
    --old training_old_version.log \
    --new training_level5_20min_final.log \
    --output figures/episode920_comparison \
    --zoom
```

**生成圖表**:
- `episode920_comparison.pdf`: 主對比圖（舊版 vs 新版）
- `episode920_zoom.pdf`: Episode 920 附近的放大圖

**論文中的使用**:

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/episode920_comparison.pdf}
    \caption{Training loss comparison at Episode 920. (a) Baseline method
    suffers numerical explosion with loss exceeding $10^6$. (b) Our
    stability-enhanced method maintains loss below 10 throughout training.}
    \label{fig:episode920}
\end{figure}
```

---

### 4. Learning Curves（標準 RL 圖表）

**腳本**: `scripts/plot_learning_curves.py`

生成標準的強化學習訓練曲線，展示性能提升。

**基本用法**:

```bash
# 單一方法
python scripts/plot_learning_curves.py \
    --data training_level5_20min_final.log \
    --labels "Ours" \
    --output figures/learning_curve

# 多方法對比
python scripts/plot_learning_curves.py \
    --data ours.log baseline1.log baseline2.log \
    --labels "Ours" "Baseline 1" "Baseline 2" \
    --output figures/learning_curve_comparison

# 生成多指標圖
python scripts/plot_learning_curves.py \
    --data training_level5_20min_final.log \
    --output figures/learning_curve \
    --multi-metric \
    --convergence
```

**生成圖表**:
- `learning_curve.pdf`: 主學習曲線（Reward vs Episode）
- `multi_metric_curves.pdf`: 多指標圖（Reward + Loss + Handovers）
- `convergence_analysis.pdf`: 收斂性分析

**參數**:
- `--smooth N`: 平滑窗口大小（預設 10）
- `--no-std`: 不顯示標準差區域
- `--multi-metric`: 生成多指標圖
- `--convergence`: 生成收斂性分析

---

### 5. 性能對比表格

**腳本**: `scripts/generate_performance_table.py`

生成論文級的性能對比表格（LaTeX / Markdown）。

**LaTeX 表格**:

```bash
python scripts/generate_performance_table.py \
    --data ours.log baseline1.log baseline2.log \
    --labels "Ours" "Baseline 1" "Baseline 2" \
    --output tables/performance_comparison.tex \
    --format latex \
    --caption "Performance comparison on LEO satellite handover task."
```

**Markdown 表格** (用於 README):

```bash
python scripts/generate_performance_table.py \
    --data ours.log \
    --labels "Ours" \
    --output tables/performance_comparison.md \
    --format markdown
```

**Ablation Study 表格**:

```bash
python scripts/generate_performance_table.py \
    --data full.log no_layer1.log no_layer2.log \
    --labels "Full Method" "w/o Layer 1" "w/o Layer 2" \
    --output tables/ablation_study.tex \
    --ablation \
    --baseline-idx 0
```

**論文中的使用**:

直接將生成的 LaTeX 程式碼貼入論文：

```latex
% 確保 preamble 中有: \usepackage{booktabs}
\input{tables/performance_comparison.tex}
```

---

### 6. Handover 分析圖（領域特定）

**腳本**: `scripts/plot_handover_analysis.py`

展示衛星切換策略的學習過程。

**用法**:

```bash
# 生成所有 Handover 分析圖
python scripts/plot_handover_analysis.py \
    --data training_level5_20min_final.log \
    --output figures/handover_analysis

# 生成綜合分析圖（2x2 子圖）
python scripts/plot_handover_analysis.py \
    --data training_level5_20min_final.log \
    --output figures/handover_comprehensive \
    --comprehensive
```

**生成圖表**:
- `handover_analysis.pdf`: Handover 頻率趨勢
- `reward_vs_handovers.pdf`: Reward vs Handovers 散點圖
- `handover_distribution.pdf`: Handover 分佈（訓練各階段）
- `handover_comprehensive.pdf`: 綜合分析圖（2x2 子圖）

---

## 📐 論文中的圖表布局建議

### 推薦的圖表順序

#### 1. Introduction / Motivation

**無圖表**（可選：問題示意圖）

#### 2. Related Work

**無圖表**（可選：方法對比表格）

#### 3. Method

**圖表建議**:
- 系統架構圖（手繪或 draw.io）
- 算法流程圖

#### 4. Experiments

這是圖表的主要部分：

**4.1 Experimental Setup**

**Table 1**: Training Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | DQN |
| Episodes | 1700 |
| Episode Duration | 20 min (240 steps) |
| Training Steps | 408,000 |
| Parallel Envs | 30 |
| Learning Rate | 2e-5 |

**4.2 Numerical Stability Analysis** ⭐ 核心貢獻

**Figure 1**: Episode 920 Comparison
- 使用: `episode920_comparison.pdf`
- 說明: 舊版本數值爆炸 vs 新版本穩定

**Figure 2**: Episode 920 Zoom-in
- 使用: `episode920_zoom.pdf`
- 說明: 詳細展示 Episode 920 前後的穩定性

**4.3 Learning Performance**

**Figure 3**: Learning Curve
- 使用: `learning_curve.pdf`
- 說明: Episode Reward 隨訓練進度提升

**Figure 4**: Multi-Metric Analysis
- 使用: `multi_metric_curves.pdf`
- 說明: Reward, Loss, Handovers 的綜合分析

**Table 2**: Performance Comparison
- 使用: `performance_comparison.tex`
- 說明: 與 Baseline 的數值對比

**4.4 Domain-Specific Analysis**

**Figure 5**: Handover Strategy Analysis
- 使用: `handover_comprehensive.pdf`
- 說明: 切換策略的學習過程

#### 5. Discussion

**可選圖表**:
- 收斂性分析
- 訓練時間對比

#### 6. Conclusion

**無圖表**

---

## 🎨 圖表樣式定制

### 修改圖表樣式

編輯 `scripts/paper_style.py` 中的配置：

```python
# 修改字體大小
setup_paper_style('neurips', font_scale=1.2)  # 增大 20%

# 修改配色
COLORS['primary'] = '#0066CC'  # 自定義藍色

# 修改圖表尺寸
fig, ax = plt.subplots(figsize=get_figure_size(width_ratio=1.5))
```

### 支援的圖表格式

所有圖表預設生成兩種格式：
- **PDF**: 用於論文（vector 格式，可無損縮放）
- **PNG**: 用於演講投影片（300 DPI）

如需其他格式：

```python
from scripts.paper_style import save_figure

save_figure(fig, 'my_figure', formats=['pdf', 'png', 'svg', 'eps'])
```

---

## 💡 最佳實踐

### 1. 訓練完成後立即生成圖表

```bash
# 訓練完成
Training completed: 1700/1700 episodes

# 立即生成圖表
./generate_paper_figures.sh
```

### 2. 定期檢查圖表品質

```bash
# 檢視 PDF 檔案
evince figures/episode920_comparison.pdf

# 或使用任何 PDF 閱讀器
```

### 3. 與 Baseline 對比

如果有多個方法需要對比：

```bash
# 生成對比學習曲線
python scripts/plot_learning_curves.py \
    --data ours.log baseline1.log baseline2.log \
    --labels "Ours (DQN + Stability)" "Baseline DQN" "Random" \
    --output figures/comparison

# 生成對比表格
python scripts/generate_performance_table.py \
    --data ours.log baseline1.log baseline2.log \
    --labels "Ours" "Baseline DQN" "Random" \
    --output tables/comparison.tex
```

### 4. 圖表說明文字（Caption）編寫建議

**好的 Caption 範例**:

```latex
\caption{Training loss comparison at Episode 920. (a) Baseline method
experiences numerical instability with loss exceeding $10^6$ at Episode 920,
preventing further training. (b) Our stability-enhanced method with 4-layer
numerical protection maintains loss below 10 throughout 1700 episodes,
demonstrating robust convergence. Shaded areas represent standard deviation
across 30 parallel environments.}
```

**Caption 應包含**:
- 圖表顯示什麼（What）
- 主要觀察結果（Key findings）
- 技術細節（如：標準差、樣本數）
- 子圖說明（如果有多個子圖）

### 5. 圖表引用方式

在論文正文中：

```latex
As shown in Figure~\ref{fig:episode920}, our method maintains numerical
stability throughout training, with loss remaining below 10 even at the
critical Episode 920 checkpoint where the baseline method fails.

The learning curve (Figure~\ref{fig:learning_curve}) demonstrates consistent
improvement, with final episode reward of $7.2 \pm 2.1$, significantly
outperforming the baseline ($1.5 \pm 1.8$, Table~\ref{tab:performance}).
```

---

## 🔧 常見問題

### Q1: 訓練還在進行中，可以生成圖表嗎？

**A**: 可以！系統會自動使用目前已完成的數據：

```bash
# 生成當前進度的圖表
./generate_paper_figures.sh --quick

# 或單獨生成學習曲線
python scripts/plot_learning_curves.py \
    --data training_level5_20min_final.log \
    --output figures/learning_curve_partial
```

### Q2: 如何添加更多 Baseline 對比？

**A**: 準備多個訓練日誌，然後：

```bash
python scripts/plot_learning_curves.py \
    --data ours.log baseline1.log baseline2.log baseline3.log \
    --labels "Ours" "DQN" "PPO" "Random" \
    --output figures/multi_method_comparison
```

### Q3: 圖表字體太小/太大？

**A**: 調整字體縮放：

```python
# 在腳本開頭修改
setup_paper_style('neurips', font_scale=1.2)  # 增大 20%
setup_paper_style('neurips', font_scale=0.8)  # 縮小 20%
```

或編輯 `scripts/paper_style.py` 中的 `base_fontsize`。

### Q4: 如何改變圖表配色？

**A**: 編輯 `scripts/paper_style.py` 中的 `COLORS` 字典：

```python
COLORS = {
    'primary': '#1f77b4',      # 改成你想要的顏色
    'secondary': '#ff7f0e',
    ...
}
```

### Q5: Episode 920 還沒到達怎麼辦？

**A**: 系統會自動處理：
- 如果訓練還沒到 Episode 920，圖表會標註預期位置
- Episode 920 放大圖會提示"尚未到達"
- 可以先生成其他圖表

### Q6: 如何生成 Ablation Study 表格？

**A**: 準備多個實驗版本的日誌：

```bash
python scripts/generate_performance_table.py \
    --data full_method.log no_layer1.log no_layer2.log no_layer3.log \
    --labels "Full Method" "w/o Layer 1" "w/o Layer 2" "w/o Layer 3" \
    --output tables/ablation_study.tex \
    --ablation \
    --baseline-idx 0  # Full Method 作為基準
```

### Q7: 如何確保圖表符合期刊要求？

**A**: 不同期刊有不同要求，常見的：

**IEEE**:
```python
setup_paper_style('ieee')  # 使用 IEEE 樣式
```

**Nature**:
```python
setup_paper_style('nature')
```

**NeurIPS/ICML/ICLR**:
```python
setup_paper_style('neurips')  # 預設推薦
```

### Q8: 生成的圖表太大/太小？

**A**: 調整圖表尺寸：

```python
# 在各腳本中修改
fig, ax = plt.subplots(figsize=get_figure_size(
    width_ratio=1.5,    # 寬度 1.5 倍
    height_ratio=0.8    # 高度 0.8 倍
))
```

---

## 📚 進階使用

### 自動化工作流程

創建自己的工作流程腳本：

```bash
#!/bin/bash
# my_paper_workflow.sh

# 1. 等待訓練完成
while [ ! -f "training_complete.flag" ]; do
    sleep 60
done

# 2. 生成所有圖表
./generate_paper_figures.sh

# 3. 發送通知（可選）
echo "圖表生成完成！" | mail -s "Training Done" your@email.com

# 4. 備份到雲端（可選）
rsync -avz figures/ your-server:/backup/figures/
```

### 批量處理多個實驗

```bash
#!/bin/bash
# batch_generate.sh

for exp in experiment_*_log; do
    exp_name=$(basename "$exp" .log)
    ./generate_paper_figures.sh --data "$exp" --output "figures_$exp_name/"
done
```

---

## 📖 參考資源

### 學術圖表設計指南

- [Ten Simple Rules for Better Figures](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833)
- [ColorBrewer](https://colorbrewer2.org/) - 色盲友好配色
- [IEEE Author Tools](https://ieeeauthorcenter.ieee.org/create-your-ieee-article/create-graphics/)

### LaTeX 圖表插入

```latex
% 單張圖
\begin{figure}[t]
    \centering
    \includegraphics[width=0.8\linewidth]{figures/episode920_comparison.pdf}
    \caption{Your caption here.}
    \label{fig:episode920}
\end{figure}

% 並排圖
\begin{figure}[t]
    \centering
    \begin{subfigure}{0.48\linewidth}
        \includegraphics[width=\linewidth]{figures/fig1.pdf}
        \caption{Subfigure 1}
    \end{subfigure}
    \hfill
    \begin{subfigure}{0.48\linewidth}
        \includegraphics[width=\linewidth]{figures/fig2.pdf}
        \caption{Subfigure 2}
    \end{subfigure}
    \caption{Overall caption.}
    \label{fig:comparison}
\end{figure}
```

---

## ✅ 檢查清單

論文投稿前的圖表檢查：

- [ ] 所有圖表都是 vector 格式（PDF/EPS）
- [ ] 圖表解析度足夠（300 DPI）
- [ ] 字體大小適中（與正文相當）
- [ ] 配色是色盲友好的
- [ ] 所有圖表都有清晰的 Caption
- [ ] 圖表編號正確（Figure 1, 2, 3...）
- [ ] 正文中正確引用所有圖表
- [ ] 表格使用 booktabs 格式
- [ ] 數值精度一致（通常 2-3 位小數）
- [ ] 誤差帶/標準差有標註
- [ ] 軸標籤清晰且有單位
- [ ] 圖例位置合適且不遮擋數據

---

## 🎯 總結

使用本系統，您可以：

1. ✅ 一鍵生成所有論文圖表
2. ✅ 符合頂級會議/期刊標準
3. ✅ 節省大量手動繪圖時間
4. ✅ 確保圖表風格一致
5. ✅ 輕鬆更新和修改圖表

**推薦工作流程**:

```bash
# 訓練完成後
./generate_paper_figures.sh           # 生成所有圖表

# 檢查圖表品質
ls -lh figures/                       # 查看生成的圖表

# 如需修改，單獨重新生成特定圖表
python scripts/plot_episode920_comparison.py --new training.log ...

# 在論文中使用
# 直接 \includegraphics{figures/episode920_comparison.pdf}
```

祝論文撰寫順利！🎓
