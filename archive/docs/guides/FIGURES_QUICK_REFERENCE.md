# 論文圖表快速參考卡

## 🚀 最常用命令

### 一鍵生成所有圖表（訓練完成後）
```bash
./generate_paper_figures.sh
```

### 快速模式（僅重要圖表）
```bash
./generate_paper_figures.sh --quick
```

### 指定訓練日誌
```bash
./generate_paper_figures.sh --data your_training.log
```

---

## 📊 單獨生成特定圖表

### Episode 920 對比圖（最重要！）
```bash
python scripts/plot_episode920_comparison.py \
    --new training_level5_20min_final.log \
    --output figures/episode920_comparison \
    --zoom
```

### Learning Curve
```bash
python scripts/plot_learning_curves.py \
    --data training_level5_20min_final.log \
    --labels "Ours" \
    --output figures/learning_curve \
    --multi-metric
```

### Handover 分析
```bash
python scripts/plot_handover_analysis.py \
    --data training_level5_20min_final.log \
    --output figures/handover_analysis \
    --comprehensive
```

### 性能表格（LaTeX）
```bash
python scripts/generate_performance_table.py \
    --data training_level5_20min_final.log \
    --labels "Ours" \
    --output tables/performance_comparison.tex \
    --format latex
```

---

## 🔄 多方法對比

### Learning Curve 對比
```bash
python scripts/plot_learning_curves.py \
    --data ours.log baseline1.log baseline2.log \
    --labels "Ours" "Baseline 1" "Baseline 2" \
    --output figures/comparison
```

### Episode 920 對比（舊版 vs 新版）
```bash
python scripts/plot_episode920_comparison.py \
    --old training_old.log \
    --new training_new.log \
    --output figures/episode920_comparison
```

---

## 📁 輸出目錄結構

```
handover-rl/
├── figures/                          # 所有生成的圖表（PDF + PNG）
│   ├── episode920_comparison.pdf    ⭐ 核心貢獻圖
│   ├── episode920_zoom.pdf
│   ├── learning_curve.pdf
│   ├── multi_metric_curves.pdf
│   ├── convergence_analysis.pdf
│   └── handover_comprehensive.pdf
├── tables/                          # LaTeX 表格
│   ├── performance_comparison.tex
│   └── performance_comparison.md
├── data/                            # 提取的數據
│   └── training_metrics.csv
└── scripts/                         # 繪圖腳本
    ├── extract_training_data.py
    ├── paper_style.py
    ├── plot_episode920_comparison.py
    ├── plot_learning_curves.py
    ├── plot_handover_analysis.py
    └── generate_performance_table.py
```

---

## 🎨 圖表樣式

### 可用樣式
- `'default'` - 通用學術樣式（推薦）
- `'ieee'` - IEEE 期刊樣式
- `'neurips'` - NeurIPS/ICML/ICLR 樣式
- `'nature'` - Nature 期刊樣式

### 修改樣式
在各腳本中修改：
```python
setup_paper_style('neurips', font_scale=1.1)
```

---

## 📋 論文中的使用順序

### Figure 1: Episode 920 對比圖（核心）
```latex
\includegraphics{figures/episode920_comparison.pdf}
```
**Caption**: 展示數值穩定性修復的效果

### Figure 2: Learning Curve
```latex
\includegraphics{figures/learning_curve.pdf}
```
**Caption**: 展示訓練過程中的性能提升

### Figure 3: Handover 分析
```latex
\includegraphics{figures/handover_comprehensive.pdf}
```
**Caption**: 展示領域特定的策略學習

### Table 1: 性能對比
```latex
\input{tables/performance_comparison.tex}
```
**Caption**: 與 Baseline 的數值對比

---

## 🔧 故障排除

### 問題：找不到數據
```bash
# 檢查日誌檔案是否存在
ls -lh training_level5_20min_final.log

# 查看日誌內容
tail -100 training_level5_20min_final.log
```

### 問題：圖表字體太小
```python
# 在腳本中修改
setup_paper_style('neurips', font_scale=1.2)  # 增大 20%
```

### 問題：需要不同格式
```python
save_figure(fig, 'output', formats=['pdf', 'png', 'svg', 'eps'])
```

---

## 💡 最佳實踐

1. **訓練完成後立即生成**: `./generate_paper_figures.sh`
2. **定期檢查圖表品質**: 開啟 PDF 檔案檢視
3. **保持數據備份**: `rsync -avz figures/ backup/`
4. **版本控制**: `git add figures/*.pdf && git commit`

---

## 📞 獲取幫助

### 查看腳本幫助
```bash
python scripts/plot_episode920_comparison.py --help
python scripts/plot_learning_curves.py --help
python scripts/plot_handover_analysis.py --help
```

### 詳細文檔
- 完整指南: `PAPER_FIGURES_GUIDE.md`
- 可視化說明: `VISUALIZATION_GUIDE.md`

---

## ✅ 快速檢查清單

投稿前確認：
- [ ] 所有圖表是 PDF 格式
- [ ] 圖表解析度 300 DPI
- [ ] 配色是色盲友好的
- [ ] 所有圖表都有 Caption
- [ ] 正文中引用所有圖表
- [ ] 表格使用 booktabs 格式
- [ ] 數值精度一致（2-3 位小數）

---

**祝論文撰寫順利！** 🎓
