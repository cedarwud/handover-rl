#!/usr/bin/env python3
"""
實時訓練監控 Web Dashboard
使用 Flask + Plotly 提供美觀的實時監控界面

Usage:
    python scripts/realtime_dashboard.py training_level5_20min_final.log

    # 瀏覽器訪問
    http://localhost:5000
"""

import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template_string, jsonify
from collections import deque
import threading
import time


# HTML 模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Training Monitor - LEO Satellite Handover RL</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: #0f1419;
            color: #e1e8ed;
            padding: 20px;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .header h1 {
            font-size: 28px;
            margin-bottom: 10px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: #1c2938;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .stat-label {
            font-size: 12px;
            color: #8899a6;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #ffffff;
        }
        .charts {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
        }
        .chart-container {
            background: #1c2938;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .chart-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #ffffff;
        }
        .status {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }
        .status.running { background: #17bf63; color: #fff; }
        .status.waiting { background: #ffad1f; color: #000; }
        .update-time {
            text-align: center;
            color: #8899a6;
            font-size: 12px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛰️ LEO Satellite Handover Training Monitor</h1>
        <p>Real-time training metrics dashboard</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-label">Status</div>
            <div class="stat-value">
                <span class="status running" id="status">RUNNING</span>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Current Episode</div>
            <div class="stat-value" id="current-episode">-</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Progress</div>
            <div class="stat-value" id="progress">-</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Latest Reward</div>
            <div class="stat-value" id="latest-reward">-</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Latest Loss</div>
            <div class="stat-value" id="latest-loss">-</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Avg Handovers</div>
            <div class="stat-value" id="latest-handovers">-</div>
        </div>
    </div>

    <div class="charts">
        <div class="chart-container">
            <div class="chart-title">📈 Episode Reward</div>
            <div id="reward-chart"></div>
        </div>
        <div class="chart-container">
            <div class="chart-title">📉 Training Loss</div>
            <div id="loss-chart"></div>
        </div>
        <div class="chart-container">
            <div class="chart-title">🔄 Handover Frequency</div>
            <div id="handover-chart"></div>
        </div>
        <div class="chart-container">
            <div class="chart-title">📊 Training Progress</div>
            <div id="progress-chart"></div>
        </div>
    </div>

    <div class="update-time">
        Last updated: <span id="last-update">-</span> |
        Auto-refresh every 5 seconds
    </div>

    <script>
        // 圖表配置
        const chartLayout = {
            paper_bgcolor: '#1c2938',
            plot_bgcolor: '#1c2938',
            font: { color: '#e1e8ed', size: 12 },
            margin: { l: 50, r: 20, t: 20, b: 40 },
            xaxis: { gridcolor: '#2a3f5f' },
            yaxis: { gridcolor: '#2a3f5f' },
            hovermode: 'x unified'
        };

        const chartConfig = {
            responsive: true,
            displayModeBar: false
        };

        // 初始化圖表
        Plotly.newPlot('reward-chart', [], chartLayout, chartConfig);
        Plotly.newPlot('loss-chart', [], chartLayout, chartConfig);
        Plotly.newPlot('handover-chart', [], chartLayout, chartConfig);
        Plotly.newPlot('progress-chart', [], chartLayout, chartConfig);

        // 更新函數
        function updateDashboard() {
            fetch('/api/data')
                .then(response => response.json())
                .then(data => {
                    if (data.episodes.length === 0) return;

                    // 更新統計卡片
                    const latest = data.episodes[data.episodes.length - 1];
                    document.getElementById('current-episode').textContent =
                        latest.episode + ' / ' + latest.total;
                    document.getElementById('progress').textContent =
                        ((latest.episode / latest.total) * 100).toFixed(1) + '%';
                    document.getElementById('latest-reward').textContent =
                        latest.reward_mean.toFixed(2);
                    document.getElementById('latest-loss').textContent =
                        latest.loss.toFixed(4);
                    document.getElementById('latest-handovers').textContent =
                        latest.handovers_mean.toFixed(1);

                    // 更新 Reward 圖表
                    const rewardTrace = {
                        x: data.episodes.map(d => d.episode),
                        y: data.episodes.map(d => d.reward_mean),
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: 'Reward',
                        line: { color: '#667eea', width: 2 },
                        marker: { size: 6 }
                    };
                    Plotly.react('reward-chart', [rewardTrace], chartLayout, chartConfig);

                    // 更新 Loss 圖表
                    const lossTrace = {
                        x: data.episodes.map(d => d.episode),
                        y: data.episodes.map(d => d.loss),
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: 'Loss',
                        line: { color: '#f093fb', width: 2 },
                        marker: { size: 6 }
                    };
                    Plotly.react('loss-chart', [lossTrace], chartLayout, chartConfig);

                    // 更新 Handover 圖表
                    const handoverTrace = {
                        x: data.episodes.map(d => d.episode),
                        y: data.episodes.map(d => d.handovers_mean),
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: 'Handovers',
                        line: { color: '#4facfe', width: 2 },
                        marker: { size: 6 }
                    };
                    Plotly.react('handover-chart', [handoverTrace], chartLayout, chartConfig);

                    // 更新進度條
                    const progressTrace = {
                        x: data.episodes.map(d => d.episode),
                        y: data.episodes.map(d => (d.episode / d.total) * 100),
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Progress',
                        fill: 'tozeroy',
                        line: { color: '#17bf63', width: 2 }
                    };
                    Plotly.react('progress-chart', [progressTrace], chartLayout, chartConfig);

                    // 更新時間
                    document.getElementById('last-update').textContent =
                        new Date().toLocaleTimeString();
                });
        }

        // 每 5 秒更新一次
        updateDashboard();
        setInterval(updateDashboard, 5000);
    </script>
</body>
</html>
"""


class TrainingDataMonitor:
    """訓練數據監控器"""

    def __init__(self, log_file: str, max_points: int = 1000):
        self.log_file = Path(log_file)
        self.max_points = max_points
        self.data = deque(maxlen=max_points)
        self.last_position = 0
        self.pattern = re.compile(
            r'Episode\s+(\d+)/(\d+):\s+'
            r'reward=([-\d.]+)±([-\d.]+),\s+'
            r'handovers=([-\d.]+)±([-\d.]+),\s+'
            r'loss=([-\d.]+|nan|inf)'
        )

        # 初始讀取
        self.update()

        # 啟動後台更新線程
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

    def _update_loop(self):
        """後台更新循環"""
        while self.running:
            self.update()
            time.sleep(5)  # 每 5 秒更新

    def update(self):
        """更新數據"""
        if not self.log_file.exists():
            return

        with open(self.log_file, 'r') as f:
            f.seek(self.last_position)
            new_lines = f.readlines()
            self.last_position = f.tell()

        for line in new_lines:
            match = self.pattern.search(line)
            if match:
                episode = int(match.group(1))
                total = int(match.group(2))
                reward_mean = float(match.group(3))
                reward_std = float(match.group(4))
                handovers_mean = float(match.group(5))
                handovers_std = float(match.group(6))

                loss_str = match.group(7)
                try:
                    loss = float(loss_str)
                except ValueError:
                    loss = float('nan') if loss_str == 'nan' else float('inf')

                self.data.append({
                    'episode': episode,
                    'total': total,
                    'reward_mean': reward_mean,
                    'reward_std': reward_std,
                    'handovers_mean': handovers_mean,
                    'handovers_std': handovers_std,
                    'loss': loss
                })

    def get_data(self):
        """獲取當前數據"""
        return list(self.data)


# Flask 應用
app = Flask(__name__)
monitor = None


@app.route('/')
def index():
    """主頁面"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/data')
def get_data():
    """API: 獲取訓練數據"""
    data = monitor.get_data()
    return jsonify({'episodes': data})


def main():
    parser = argparse.ArgumentParser(
        description='實時訓練監控 Web Dashboard'
    )
    parser.add_argument('log_file', type=str,
                       help='訓練日誌檔案路徑')
    parser.add_argument('--port', type=int, default=5000,
                       help='Web 服務端口（預設: 5000）')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Web 服務 IP（預設: 0.0.0.0，允許外部訪問）')

    args = parser.parse_args()

    # 初始化監控器
    global monitor
    monitor = TrainingDataMonitor(args.log_file)

    print("="*70)
    print("🚀 實時訓練監控 Dashboard 已啟動")
    print("="*70)
    print(f"")
    print(f"📝 監控日誌: {args.log_file}")
    print(f"🌐 訪問地址: http://localhost:{args.port}")
    print(f"")
    print(f"💡 提示:")
    print(f"   - Dashboard 每 5 秒自動更新")
    print(f"   - 圖表會隨訓練實時變化")
    print(f"   - 按 Ctrl+C 停止服務")
    print(f"")
    print("="*70)

    # 啟動 Flask 服務
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
