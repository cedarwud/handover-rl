# 工具目錄

本目錄包含常用的訓練、監控和分析工具。

## 🔧 可用工具

### 訓練相關

#### train_level5_final.sh
最終訓練腳本（Level 5 數值穩定）
```bash
./tools/train_level5_final.sh
```

---

### 監控相關

#### check_progress.sh
快速查看訓練進度
```bash
./tools/check_progress.sh
```

顯示：
- 訓練進程狀態
- 最新完成的 episodes
- 統計信息（完成數、無效動作數）
- GPU 狀態

---

#### auto_monitor.sh
自動監控腳本（每5分鐘檢查）
```bash
./tools/auto_monitor.sh &
```

監控內容：
- 訓練進程是否運行
- Episode 進度是否卡住
- 無效動作警告
- Loss 爆炸檢測
- 里程碑提醒（Episode 10, 50, 100, 920, 1700）

日誌位於: `../logs/training_monitor.log`

---

#### view_training_log.sh
查看訓練日誌
```bash
./tools/view_training_log.sh
```

---

#### view_monitor.sh
查看監控狀態
```bash
./tools/view_monitor.sh
```

---

### 分析相關

#### analyze_training.sh
分析訓練結果
```bash
./tools/analyze_training.sh
```

生成：
- 統計摘要
- 學習曲線分析
- 異常檢測報告

---

#### generate_paper_figures.sh
生成論文圖表（6組圖+表格）
```bash
./tools/generate_paper_figures.sh
```

生成文件：
- `figures/learning_curve.pdf/png`
- `figures/multi_metric_curves.pdf/png`
- `figures/convergence_analysis.pdf/png`
- `figures/episode920_comparison.pdf/png`
- `figures/handover_analysis.pdf/png`
- `tables/performance_comparison.tex`

詳見: `../docs/PAPER_FIGURES_GUIDE.md`

---

### 前端相關

#### live_monitor.html
實時監控儀表板（HTML頁面）
```bash
# 在瀏覽器中打開
firefox ./tools/live_monitor.html
```

---

## 📋 歷史工具

更多歷史監控和測試腳本保存在：
- `../archive/scripts/monitoring/`: 舊監控腳本
- `../archive/scripts/testing/`: 舊測試腳本

---

**最後更新**: 2025-11-08
