#!/usr/bin/env python3
"""
訓練數據提取腳本
從訓練日誌中提取 episode、reward、loss 等數據用於繪圖

Usage:
    python scripts/extract_training_data.py training_level5_20min_final.log
    python scripts/extract_training_data.py training_level5_20min_final.log --output data/training_metrics.csv
"""

import re
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


def extract_episode_data(log_file: Path) -> pd.DataFrame:
    """
    從訓練日誌提取 episode 數據

    日誌格式示例:
    INFO:__main__:Episode   10/1700: reward=-1257.56±813.72, handovers=13.8±17.3, loss=14.8382
    INFO:__main__:Episode   20/1700: reward=-648.07±732.17, handovers=26.4±25.0, loss=5.2182

    Args:
        log_file: 訓練日誌檔案路徑

    Returns:
        DataFrame with columns: episode, reward_mean, reward_std, handovers_mean, handovers_std, loss
    """

    data = {
        'episode': [],
        'total_episodes': [],
        'reward_mean': [],
        'reward_std': [],
        'handovers_mean': [],
        'handovers_std': [],
        'loss': []
    }

    # 正則表達式匹配日誌行
    # Episode   10/1700: reward=-1257.56±813.72, handovers=13.8±17.3, loss=14.8382
    pattern = re.compile(
        r'Episode\s+(\d+)/(\d+):\s+'
        r'reward=([-\d.]+)±([-\d.]+),\s+'
        r'handovers=([-\d.]+)±([-\d.]+),\s+'
        r'loss=([-\d.]+|nan|inf)'
    )

    print(f"📖 讀取日誌: {log_file}")
    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            match = pattern.search(line)
            if match:
                episode = int(match.group(1))
                total = int(match.group(2))
                reward_mean = float(match.group(3))
                reward_std = float(match.group(4))
                handovers_mean = float(match.group(5))
                handovers_std = float(match.group(6))

                # 處理 loss (可能是 nan 或 inf)
                loss_str = match.group(7)
                try:
                    loss = float(loss_str)
                except ValueError:
                    loss = float('nan') if loss_str == 'nan' else float('inf')

                data['episode'].append(episode)
                data['total_episodes'].append(total)
                data['reward_mean'].append(reward_mean)
                data['reward_std'].append(reward_std)
                data['handovers_mean'].append(handovers_mean)
                data['handovers_std'].append(handovers_std)
                data['loss'].append(loss)

    df = pd.DataFrame(data)

    if len(df) == 0:
        print("⚠️  警告: 未找到任何 episode 數據")
        print(f"    請檢查日誌格式是否符合預期")
    else:
        print(f"✅ 提取成功: {len(df)} 個 episodes")
        print(f"   Episode 範圍: {df['episode'].min()} - {df['episode'].max()}")
        print(f"   Reward 範圍: {df['reward_mean'].min():.2f} - {df['reward_mean'].max():.2f}")
        print(f"   Loss 範圍: {df['loss'].min():.2f} - {df['loss'].max():.2f}")

    return df


def compute_statistics(df: pd.DataFrame) -> Dict:
    """計算訓練統計數據"""

    stats = {
        'total_episodes': len(df),
        'final_reward_mean': df['reward_mean'].iloc[-1] if len(df) > 0 else None,
        'final_reward_std': df['reward_std'].iloc[-1] if len(df) > 0 else None,
        'best_reward': df['reward_mean'].max() if len(df) > 0 else None,
        'best_reward_episode': df.loc[df['reward_mean'].idxmax(), 'episode'] if len(df) > 0 else None,
        'final_loss': df['loss'].iloc[-1] if len(df) > 0 else None,
        'min_loss': df['loss'].min() if len(df) > 0 else None,
        'max_loss': df['loss'].max() if len(df) > 0 else None,
        'avg_handovers': df['handovers_mean'].mean() if len(df) > 0 else None,
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description='從訓練日誌提取數據')
    parser.add_argument('log_file', type=str, help='訓練日誌檔案路徑')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='輸出 CSV 檔案路徑 (預設: {log_file}.csv)')
    parser.add_argument('--stats', '-s', action='store_true',
                       help='顯示統計摘要')

    args = parser.parse_args()

    # 確認日誌檔案存在
    log_file = Path(args.log_file)
    if not log_file.exists():
        print(f"❌ 錯誤: 日誌檔案不存在: {log_file}")
        return 1

    # 提取數據
    df = extract_episode_data(log_file)

    if len(df) == 0:
        return 1

    # 決定輸出路徑
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = log_file.with_suffix('.csv')

    # 確保輸出目錄存在
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 儲存 CSV
    df.to_csv(output_file, index=False)
    print(f"💾 數據已儲存: {output_file}")

    # 顯示統計摘要
    if args.stats:
        print("\n" + "="*60)
        print("📊 訓練統計摘要")
        print("="*60)

        stats = compute_statistics(df)
        print(f"總 Episodes:        {stats['total_episodes']}")
        print(f"最終 Reward:        {stats['final_reward_mean']:.2f} ± {stats['final_reward_std']:.2f}")
        print(f"最佳 Reward:        {stats['best_reward']:.2f} (Episode {stats['best_reward_episode']})")
        print(f"最終 Loss:          {stats['final_loss']:.4f}")
        print(f"最小 Loss:          {stats['min_loss']:.4f}")
        print(f"最大 Loss:          {stats['max_loss']:.4f}")
        print(f"平均 Handovers:     {stats['avg_handovers']:.2f}")
        print("="*60)

    return 0


if __name__ == '__main__':
    exit(main())
