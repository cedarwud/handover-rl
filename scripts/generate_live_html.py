#!/usr/bin/env python3
"""
自動刷新 HTML 報告生成器
定期生成靜態 HTML，瀏覽器自動刷新

Usage:
    # 生成實時 HTML 報告（每 10 秒更新）
    python scripts/generate_live_html.py training_level5_20min_final.log \
        --output live_monitor.html &

    # 用瀏覽器打開
    firefox live_monitor.html
    # 或
    chromium live_monitor.html
"""

import re
import time
import argparse
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.extract_training_data import extract_episode_data


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="10">
    <title>Training Monitor</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .header h1 {{ font-size: 32px; margin-bottom: 10px; }}
        .header .subtitle {{ font-size: 14px; opacity: 0.9; }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat {{
            background: #16213e;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .stat-label {{
            font-size: 12px;
            color: #aaa;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .chart {{
            background: #16213e;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .footer {{
            text-align: center;
            color: #888;
            font-size: 12px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛰️ Training Monitor</h1>
        <div class="subtitle">LEO Satellite Handover RL - Auto-refresh every 10 seconds</div>
    </div>

    <div class="stats">
        <div class="stat">
            <div class="stat-label">Current Episode</div>
            <div class="stat-value">{current_episode} / {total_episodes}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Progress</div>
            <div class="stat-value">{progress}%</div>
        </div>
        <div class="stat">
            <div class="stat-label">Latest Reward</div>
            <div class="stat-value">{latest_reward}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Latest Loss</div>
            <div class="stat-value">{latest_loss}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Handovers</div>
            <div class="stat-value">{latest_handovers}</div>
        </div>
    </div>

    <div class="chart" id="reward-chart"></div>
    <div class="chart" id="loss-chart"></div>
    <div class="chart" id="handover-chart"></div>

    <div class="footer">
        Last updated: {update_time}<br>
        Page will auto-refresh in 10 seconds
    </div>

    <script>
        const chartLayout = {{
            paper_bgcolor: '#16213e',
            plot_bgcolor: '#16213e',
            font: {{ color: '#eee' }},
            margin: {{ l: 50, r: 20, t: 40, b: 40 }},
            xaxis: {{ gridcolor: '#2a3f5f', title: 'Episode' }},
            yaxis: {{ gridcolor: '#2a3f5f' }},
        }};

        const chartConfig = {{ responsive: true, displayModeBar: false }};

        // Reward 圖表
        Plotly.newPlot('reward-chart', [{{
            x: {episodes_json},
            y: {rewards_json},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Reward',
            line: {{ color: '#667eea', width: 2 }},
            marker: {{ size: 6 }}
        }}], {{
            ...chartLayout,
            title: '📈 Episode Reward',
            yaxis: {{ ...chartLayout.yaxis, title: 'Reward' }}
        }}, chartConfig);

        // Loss 圖表
        Plotly.newPlot('loss-chart', [{{
            x: {episodes_json},
            y: {losses_json},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Loss',
            line: {{ color: '#f093fb', width: 2 }},
            marker: {{ size: 6 }}
        }}], {{
            ...chartLayout,
            title: '📉 Training Loss',
            yaxis: {{ ...chartLayout.yaxis, title: 'Loss' }}
        }}, chartConfig);

        // Handover 圖表
        Plotly.newPlot('handover-chart', [{{
            x: {episodes_json},
            y: {handovers_json},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Handovers',
            line: {{ color: '#4facfe', width: 2 }},
            marker: {{ size: 6 }}
        }}], {{
            ...chartLayout,
            title: '🔄 Handover Frequency',
            yaxis: {{ ...chartLayout.yaxis, title: 'Handovers per Episode' }}
        }}, chartConfig);
    </script>
</body>
</html>
"""


def generate_html_report(log_file: Path, output_file: Path):
    """生成 HTML 報告"""

    # 提取數據
    data = extract_episode_data(log_file)

    if len(data) == 0:
        # 無數據時的 HTML
        html = HTML_TEMPLATE.format(
            current_episode=0,
            total_episodes=0,
            progress=0.0,
            latest_reward="N/A",
            latest_loss="N/A",
            latest_handovers="N/A",
            episodes_json="[]",
            rewards_json="[]",
            losses_json="[]",
            handovers_json="[]",
            update_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
    else:
        # 準備數據
        latest = data.iloc[-1]

        episodes = data['episode'].tolist()
        rewards = data['reward_mean'].tolist()
        losses = data['loss'].tolist()
        handovers = data['handovers_mean'].tolist()

        # 生成 HTML
        html = HTML_TEMPLATE.format(
            current_episode=int(latest['episode']),
            total_episodes=int(latest['total_episodes']),
            progress=round((latest['episode'] / latest['total_episodes']) * 100, 1),
            latest_reward=f"{latest['reward_mean']:.2f}±{latest['reward_std']:.2f}",
            latest_loss=f"{latest['loss']:.4f}",
            latest_handovers=f"{latest['handovers_mean']:.1f}±{latest['handovers_std']:.1f}",
            episodes_json=episodes,
            rewards_json=rewards,
            losses_json=losses,
            handovers_json=handovers,
            update_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )

    # 寫入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(
        description='自動刷新 HTML 報告生成器'
    )
    parser.add_argument('log_file', type=str,
                       help='訓練日誌檔案路徑')
    parser.add_argument('--output', '-o', type=str,
                       default='live_monitor.html',
                       help='輸出 HTML 檔案路徑（預設: live_monitor.html）')
    parser.add_argument('--interval', type=int, default=10,
                       help='更新間隔秒數（預設: 10）')
    parser.add_argument('--once', action='store_true',
                       help='只生成一次，不持續更新')

    args = parser.parse_args()

    log_file = Path(args.log_file)
    output_file = Path(args.output)

    if not log_file.exists():
        print(f"❌ 錯誤: 日誌檔案不存在: {log_file}")
        return 1

    print("="*70)
    print("📊 實時 HTML 報告生成器")
    print("="*70)
    print(f"")
    print(f"📝 監控日誌: {log_file}")
    print(f"📄 輸出檔案: {output_file.absolute()}")
    print(f"🔄 更新間隔: {args.interval} 秒")
    print(f"")
    print(f"💡 使用方法:")
    print(f"   1. 用瀏覽器打開: {output_file.absolute()}")
    print(f"   2. 頁面會每 10 秒自動刷新")
    print(f"   3. 按 Ctrl+C 停止生成")
    print(f"")
    print("="*70)
    print(f"")

    try:
        if args.once:
            # 只生成一次
            print(f"📊 生成報告...")
            generate_html_report(log_file, output_file)
            print(f"✅ 報告已生成: {output_file.absolute()}")
        else:
            # 持續更新
            print(f"🔄 開始持續更新...")
            while True:
                generate_html_report(log_file, output_file)
                print(f"✅ [{datetime.now().strftime('%H:%M:%S')}] 報告已更新")
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\n\n⏹️  生成器已停止")
        print(f"📄 最終報告: {output_file.absolute()}")

    return 0


if __name__ == '__main__':
    exit(main())
