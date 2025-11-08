#!/bin/bash
#========================================
# 實時監控啟動腳本
#========================================
# 快速啟動各種實時監控方案
#
# Usage:
#   ./start_monitor.sh tensorboard
#   ./start_monitor.sh dashboard
#   ./start_monitor.sh html

set -e

# 預設訓練日誌
LOG_FILE="training_level5_20min_final.log"

# 顏色輸出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 檢查虛擬環境
if [ -z "$VIRTUAL_ENV" ]; then
    echo "🔧 啟動虛擬環境..."
    source venv/bin/activate
fi

# 解析參數
MODE="$1"

if [ -z "$MODE" ]; then
    echo "========================================"
    echo "🚀 實時監控啟動器"
    echo "========================================"
    echo ""
    echo "請選擇監控方案:"
    echo ""
    echo "  1. tensorboard  - TensorBoard 監控（RL 標準，推薦）"
    echo "  2. dashboard    - Web Dashboard（美觀，互動式）"
    echo "  3. html         - 自動刷新 HTML（無需額外服務）"
    echo ""
    echo "Usage:"
    echo "  ./start_monitor.sh tensorboard"
    echo "  ./start_monitor.sh dashboard"
    echo "  ./start_monitor.sh html"
    echo ""
    exit 0
fi

# 檢查日誌檔案
if [ ! -f "$LOG_FILE" ]; then
    echo -e "${YELLOW}⚠️  警告: 訓練日誌不存在: $LOG_FILE${NC}"
    echo "   請確認訓練已開始，或修改 LOG_FILE 變數"
    exit 1
fi

case "$MODE" in
    tensorboard|tb)
        echo "========================================"
        echo "🚀 啟動 TensorBoard 監控"
        echo "========================================"
        echo ""
        echo "📝 監控日誌: $LOG_FILE"
        echo ""
        echo "步驟 1/2: 啟動數據轉換器..."
        python3 scripts/realtime_tensorboard.py "$LOG_FILE" &
        TB_PID=$!
        sleep 3
        echo -e "${GREEN}✅ 數據轉換器已啟動 (PID: $TB_PID)${NC}"
        echo ""
        echo "步驟 2/2: 啟動 TensorBoard..."
        echo ""
        echo -e "${BLUE}🌐 TensorBoard 將在瀏覽器打開...${NC}"
        echo -e "${BLUE}   訪問地址: http://localhost:6006${NC}"
        echo ""
        tensorboard --logdir=logs/tensorboard --port=6006
        ;;

    dashboard|web)
        echo "========================================"
        echo "🚀 啟動 Web Dashboard"
        echo "========================================"
        echo ""
        echo "📝 監控日誌: $LOG_FILE"
        echo ""
        echo -e "${GREEN}正在啟動 Flask 服務...${NC}"
        echo ""
        python3 scripts/realtime_dashboard.py "$LOG_FILE"
        ;;

    html|static)
        echo "========================================"
        echo "🚀 啟動自動刷新 HTML 報告"
        echo "========================================"
        echo ""
        echo "📝 監控日誌: $LOG_FILE"
        echo "📄 輸出檔案: live_monitor.html"
        echo ""
        echo -e "${GREEN}正在生成報告...${NC}"
        python3 scripts/generate_live_html.py "$LOG_FILE" &
        HTML_PID=$!
        sleep 2
        echo ""
        echo -e "${GREEN}✅ 報告生成器已啟動 (PID: $HTML_PID)${NC}"
        echo ""
        echo -e "${BLUE}🌐 用瀏覽器打開: file://$(pwd)/live_monitor.html${NC}"
        echo ""
        echo "提示:"
        echo "  - 頁面每 10 秒自動刷新"
        echo "  - 按 Ctrl+C 停止生成器"
        echo ""
        wait $HTML_PID
        ;;

    *)
        echo "❌ 錯誤: 未知模式 '$MODE'"
        echo ""
        echo "可用模式:"
        echo "  tensorboard - TensorBoard 監控"
        echo "  dashboard   - Web Dashboard"
        echo "  html        - 自動刷新 HTML"
        exit 1
        ;;
esac
