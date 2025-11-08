#!/bin/bash
#========================================
# 論文圖表生成主控腳本
#========================================
# 自動生成所有論文級圖表和表格
#
# Usage:
#   ./generate_paper_figures.sh                    # 生成所有圖表
#   ./generate_paper_figures.sh --quick            # 僅生成最重要的圖表
#   ./generate_paper_figures.sh --data mylog.log  # 指定訓練日誌

set -e  # 遇到錯誤立即退出

# ========================================
# 配置
# ========================================
DEFAULT_DATA_FILE="training_level5_20min_final.log"
OUTPUT_DIR="figures"
TABLES_DIR="tables"
DATA_DIR="data"

# 解析參數
QUICK_MODE=false
DATA_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --data)
            DATA_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick          僅生成最重要的圖表（Episode 920 + Learning Curve）"
            echo "  --data FILE      指定訓練日誌檔案（預設: $DEFAULT_DATA_FILE）"
            echo "  --help, -h       顯示此幫助訊息"
            echo ""
            echo "輸出位置:"
            echo "  圖表: $OUTPUT_DIR/"
            echo "  表格: $TABLES_DIR/"
            echo "  數據: $DATA_DIR/"
            exit 0
            ;;
        *)
            echo "未知參數: $1"
            echo "使用 --help 查看幫助"
            exit 1
            ;;
    esac
done

# 設定數據檔案
if [ -z "$DATA_FILE" ]; then
    DATA_FILE="$DEFAULT_DATA_FILE"
fi

# ========================================
# 環境檢查
# ========================================
echo "========================================"
echo "論文圖表生成系統"
echo "========================================"
echo ""
echo "⚙️  配置:"
echo "   訓練日誌: $DATA_FILE"
echo "   輸出目錄: $OUTPUT_DIR/"
echo "   快速模式: $QUICK_MODE"
echo ""

# 檢查數據檔案是否存在
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ 錯誤: 訓練日誌不存在: $DATA_FILE"
    echo ""
    echo "💡 請確認："
    echo "   1. 訓練是否已開始？"
    echo "   2. 日誌檔案路徑是否正確？"
    echo "   3. 使用 --data 參數指定正確的日誌檔案"
    exit 1
fi

# 檢查虛擬環境
if [ -z "$VIRTUAL_ENV" ]; then
    echo "🔧 啟動虛擬環境..."
    source venv/bin/activate
fi

# 創建輸出目錄
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TABLES_DIR"
mkdir -p "$DATA_DIR"

# ========================================
# 步驟 1: 提取訓練數據
# ========================================
echo ""
echo "========================================"
echo "📊 步驟 1/5: 提取訓練數據"
echo "========================================"
python3 scripts/extract_training_data.py \
    "$DATA_FILE" \
    --output "$DATA_DIR/training_metrics.csv" \
    --stats

# ========================================
# 步驟 2: Episode 920 對比圖（最重要）
# ========================================
echo ""
echo "========================================"
echo "🎨 步驟 2/5: Episode 920 對比圖"
echo "========================================"
echo "這是論文中最重要的圖表（核心技術貢獻）"

python3 scripts/plot_episode920_comparison.py \
    --new "$DATA_FILE" \
    --output "$OUTPUT_DIR/episode920_comparison" \
    --zoom

echo "✅ Episode 920 圖表完成"

# ========================================
# 步驟 3: Learning Curves（標準 RL 圖表）
# ========================================
echo ""
echo "========================================"
echo "🎨 步驟 3/5: Learning Curves"
echo "========================================"

python3 scripts/plot_learning_curves.py \
    --data "$DATA_FILE" \
    --labels "Ours" \
    --output "$OUTPUT_DIR/learning_curve" \
    --smooth 10 \
    --multi-metric \
    --convergence

echo "✅ Learning Curves 完成"

# 快速模式：只生成前三個最重要的圖表
if [ "$QUICK_MODE" = true ]; then
    echo ""
    echo "========================================"
    echo "✅ 快速模式完成！"
    echo "========================================"
    echo ""
    echo "已生成最重要的圖表："
    echo "  1. Episode 920 對比圖: $OUTPUT_DIR/episode920_comparison.pdf"
    echo "  2. Episode 920 放大圖: $OUTPUT_DIR/episode920_zoom.pdf"
    echo "  3. Learning Curve: $OUTPUT_DIR/learning_curve.pdf"
    echo "  4. 多指標圖: $OUTPUT_DIR/multi_metric_curves.pdf"
    echo "  5. 收斂性分析: $OUTPUT_DIR/convergence_analysis.pdf"
    echo ""
    echo "💡 使用 ./generate_paper_figures.sh 生成完整圖表集"
    exit 0
fi

# ========================================
# 步驟 4: Handover 分析圖（領域特定）
# ========================================
echo ""
echo "========================================"
echo "🎨 步驟 4/5: Handover 分析"
echo "========================================"

python3 scripts/plot_handover_analysis.py \
    --data "$DATA_FILE" \
    --output "$OUTPUT_DIR/handover_analysis" \
    --smooth 10 \
    --comprehensive

echo "✅ Handover 分析完成"

# ========================================
# 步驟 5: 性能對比表格
# ========================================
echo ""
echo "========================================"
echo "📋 步驟 5/5: 性能對比表格"
echo "========================================"

# LaTeX 表格
python3 scripts/generate_performance_table.py \
    --data "$DATA_FILE" \
    --labels "Ours" \
    --output "$TABLES_DIR/performance_comparison.tex" \
    --format latex \
    --caption "Performance of our method on LEO satellite handover task."

# Markdown 表格（用於 README）
python3 scripts/generate_performance_table.py \
    --data "$DATA_FILE" \
    --labels "Ours" \
    --output "$TABLES_DIR/performance_comparison.md" \
    --format markdown

echo "✅ 性能表格完成"

# ========================================
# 完成總結
# ========================================
echo ""
echo "========================================"
echo "🎉 所有論文圖表生成完成！"
echo "========================================"
echo ""
echo "📊 生成的圖表："
echo ""
echo "【核心技術貢獻】"
echo "  ✅ Episode 920 對比圖: $OUTPUT_DIR/episode920_comparison.pdf"
echo "  ✅ Episode 920 放大圖: $OUTPUT_DIR/episode920_zoom.pdf"
echo ""
echo "【標準 RL 圖表】"
echo "  ✅ Learning Curve: $OUTPUT_DIR/learning_curve.pdf"
echo "  ✅ 多指標曲線: $OUTPUT_DIR/multi_metric_curves.pdf"
echo "  ✅ 收斂性分析: $OUTPUT_DIR/convergence_analysis.pdf"
echo ""
echo "【領域特定分析】"
echo "  ✅ Handover 綜合分析: $OUTPUT_DIR/handover_comprehensive.pdf"
echo ""
echo "📋 生成的表格："
echo "  ✅ LaTeX 表格: $TABLES_DIR/performance_comparison.tex"
echo "  ✅ Markdown 表格: $TABLES_DIR/performance_comparison.md"
echo ""
echo "📁 提取的數據："
echo "  ✅ 訓練指標: $DATA_DIR/training_metrics.csv"
echo ""
echo "========================================"
echo "💡 使用建議"
echo "========================================"
echo ""
echo "1. 論文中的圖表使用順序："
echo "   - Figure 1: Episode 920 對比圖 (核心貢獻)"
echo "   - Figure 2: Learning Curve (性能展示)"
echo "   - Figure 3: Handover 分析 (領域特定)"
echo "   - Table 1: 性能對比表格"
echo ""
echo "2. 圖表說明文字範例："
echo "   - 參考各腳本生成時的 Caption 建議"
echo "   - 強調數值穩定性改進"
echo "   - 說明訓練量的學術正當性"
echo ""
echo "3. 如需與 Baseline 對比："
echo "   - 重新運行各腳本並提供多個日誌檔案"
echo "   - 例: --data ours.log baseline.log"
echo ""
echo "4. 查看圖表："
echo "   - PDF 檔案可直接在論文中使用"
echo "   - PNG 檔案用於演講投影片"
echo ""
echo "========================================"
