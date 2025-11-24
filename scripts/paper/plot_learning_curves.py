#!/usr/bin/env python3
"""
Learning Curves 生成器 - 標準 RL 論文圖表
展示訓練過程中的性能提升

符合 NeurIPS / ICML / ICLR 標準格式

Usage:
    # 單一方法的學習曲線
    python scripts/plot_learning_curves.py \\
        --data training_level5_20min_final.log \\
        --output figures/learning_curve

    # 多個方法對比
    python scripts/plot_learning_curves.py \\
        --data method1.log method2.log method3.log \\
        --labels "Ours" "Baseline 1" "Baseline 2" \\
        --output figures/learning_curve_comparison
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# 導入樣式配置
script_dir = Path(__file__).parent.parent  # scripts/
sys.path.insert(0, str(script_dir))
from paper.paper_style import (
    setup_paper_style, get_figure_size, save_figure,
    COLORS, COLOR_PALETTE, MARKERS
)
from extract_training_data import extract_episode_data


def smooth_curve(data: np.ndarray, window_size: int = 10) -> np.ndarray:
    """
    平滑曲線（使用移動平均）

    Args:
        data: 原始數據
        window_size: 窗口大小

    Returns:
        平滑後的數據
    """
    if len(data) < window_size:
        return data

    return uniform_filter1d(data, size=window_size, mode='nearest')


def plot_learning_curve(data_list: list,
                       labels: list = None,
                       output_file: str = 'figures/learning_curve',
                       smooth_window: int = 10,
                       show_std: bool = True,
                       x_axis: str = 'episode'):
    """
    生成 Learning Curve 圖表

    Args:
        data_list: 數據列表 (list of DataFrames)
        labels: 標籤列表
        output_file: 輸出檔案路徑
        smooth_window: 平滑窗口大小
        show_std: 是否顯示標準差區域
        x_axis: X 軸類型 ('episode' 或 'timestep')

    Returns:
        fig, ax: matplotlib Figure 和 Axes 物件
    """

    setup_paper_style('neurips', font_scale=1.1)

    fig, ax = plt.subplots(figsize=get_figure_size())

    if labels is None:
        labels = [f'Method {i+1}' for i in range(len(data_list))]

    # 繪製每個方法的學習曲線
    for idx, (data, label) in enumerate(zip(data_list, labels)):
        episodes = data['episode'].values
        reward_mean = data['reward_mean'].values
        reward_std = data['reward_std'].values if show_std else None

        # 平滑曲線
        if smooth_window > 1:
            reward_mean_smooth = smooth_curve(reward_mean, smooth_window)
            if reward_std is not None:
                reward_std_smooth = smooth_curve(reward_std, smooth_window)
        else:
            reward_mean_smooth = reward_mean
            reward_std_smooth = reward_std

        # 選擇顏色和樣式
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        marker = MARKERS[idx % len(MARKERS)]

        # 繪製主曲線
        ax.plot(episodes, reward_mean_smooth,
               color=color,
               linewidth=2.5,
               label=label,
               marker=marker,
               markersize=4,
               markevery=max(1, len(episodes) // 15),
               zorder=10 - idx)

        # 繪製標準差區域
        if show_std and reward_std_smooth is not None:
            ax.fill_between(episodes,
                           reward_mean_smooth - reward_std_smooth,
                           reward_mean_smooth + reward_std_smooth,
                           color=color,
                           alpha=0.2,
                           zorder=10 - idx - 0.5)

    # 設定軸標籤和標題
    xlabel = 'Episode' if x_axis == 'episode' else 'Training Steps'
    ax.set_xlabel(xlabel, fontsize=12, weight='bold')
    ax.set_ylabel('Episode Reward', fontsize=12, weight='bold')
    ax.set_title('Learning Curve', fontsize=13, weight='bold')

    # 圖例
    ax.legend(loc='best', framealpha=0.95, fontsize=10)

    # 網格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # 設定 X 軸起點為 0
    ax.set_xlim(left=0)

    plt.tight_layout()
    save_figure(fig, output_file, formats=['pdf', 'png'])

    return fig, ax


def plot_multi_metric_curves(data: pd.DataFrame,
                            output_file: str = 'figures/multi_metric_curves',
                            smooth_window: int = 10):
    """
    生成多指標學習曲線（Reward + Loss + Handovers）

    Args:
        data: 訓練數據 (DataFrame)
        output_file: 輸出檔案路徑
        smooth_window: 平滑窗口大小

    Returns:
        fig, axes: matplotlib Figure 和 Axes 物件
    """

    setup_paper_style('neurips', font_scale=1.0)

    fig, axes = plt.subplots(3, 1, figsize=get_figure_size(height_ratio=1.2))

    episodes = data['episode'].values

    # ========================================
    # 子圖 1: Episode Reward
    # ========================================
    reward_mean = data['reward_mean'].values
    reward_std = data['reward_std'].values

    if smooth_window > 1:
        reward_mean_smooth = smooth_curve(reward_mean, smooth_window)
        reward_std_smooth = smooth_curve(reward_std, smooth_window)
    else:
        reward_mean_smooth = reward_mean
        reward_std_smooth = reward_std

    axes[0].plot(episodes, reward_mean_smooth,
                color=COLORS['primary'],
                linewidth=2.0,
                label='Mean Reward')

    axes[0].fill_between(episodes,
                         reward_mean_smooth - reward_std_smooth,
                         reward_mean_smooth + reward_std_smooth,
                         color=COLORS['primary'],
                         alpha=0.2,
                         label='±σ')

    axes[0].set_ylabel('Episode Reward', fontsize=10, weight='bold')
    axes[0].set_title('(a) Learning Progress: Episode Reward', fontsize=11, weight='bold')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # ========================================
    # 子圖 2: Training Loss
    # ========================================
    loss = data['loss'].values

    if smooth_window > 1:
        loss_smooth = smooth_curve(loss, smooth_window)
    else:
        loss_smooth = loss

    axes[1].plot(episodes, loss_smooth,
                color=COLORS['secondary'],
                linewidth=2.0,
                label='Training Loss')

    # 添加穩定性閾值線
    axes[1].axhline(10, color=COLORS['danger'], linestyle='--',
                   linewidth=1.5, alpha=0.5, label='Stability threshold')

    axes[1].set_ylabel('Training Loss', fontsize=10, weight='bold')
    axes[1].set_title('(b) Training Stability: Loss', fontsize=11, weight='bold')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # ========================================
    # 子圖 3: Handover Frequency
    # ========================================
    handovers_mean = data['handovers_mean'].values
    handovers_std = data['handovers_std'].values

    if smooth_window > 1:
        handovers_mean_smooth = smooth_curve(handovers_mean, smooth_window)
        handovers_std_smooth = smooth_curve(handovers_std, smooth_window)
    else:
        handovers_mean_smooth = handovers_mean
        handovers_std_smooth = handovers_std

    axes[2].plot(episodes, handovers_mean_smooth,
                color=COLORS['tertiary'],
                linewidth=2.0,
                label='Mean Handovers')

    axes[2].fill_between(episodes,
                         handovers_mean_smooth - handovers_std_smooth,
                         handovers_mean_smooth + handovers_std_smooth,
                         color=COLORS['tertiary'],
                         alpha=0.2,
                         label='±σ')

    axes[2].set_xlabel('Episode', fontsize=10, weight='bold')
    axes[2].set_ylabel('Handovers per Episode', fontsize=10, weight='bold')
    axes[2].set_title('(c) Handover Strategy: Frequency', fontsize=11, weight='bold')
    axes[2].legend(loc='best', fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_file, formats=['pdf', 'png'])

    return fig, axes


def plot_convergence_analysis(data: pd.DataFrame,
                              output_file: str = 'figures/convergence_analysis',
                              convergence_threshold: float = 0.1):
    """
    收斂性分析圖

    Args:
        data: 訓練數據
        output_file: 輸出檔案路徑
        convergence_threshold: 收斂閾值（reward 變化率）

    Returns:
        fig, axes: matplotlib Figure 和 Axes 物件
    """

    setup_paper_style('neurips', font_scale=1.0)

    fig, axes = plt.subplots(2, 1, figsize=get_figure_size(height_ratio=0.8))

    episodes = data['episode'].values
    reward_mean = data['reward_mean'].values

    # 計算收斂性指標
    # 1. 移動平均差異 (判斷是否收斂)
    window = 50
    if len(reward_mean) >= window * 2:
        moving_avg = smooth_curve(reward_mean, window)
        moving_std = np.array([
            np.std(reward_mean[max(0, i-window):i+window])
            for i in range(len(reward_mean))
        ])

        # 子圖 1: Reward with moving average
        axes[0].plot(episodes, reward_mean,
                    color=COLORS['primary'],
                    alpha=0.3,
                    linewidth=1.0,
                    label='Raw Reward')

        axes[0].plot(episodes, moving_avg,
                    color=COLORS['primary'],
                    linewidth=2.5,
                    label=f'Moving Avg (window={window})')

        axes[0].set_ylabel('Episode Reward', fontsize=10, weight='bold')
        axes[0].set_title('(a) Reward Convergence', fontsize=11, weight='bold')
        axes[0].legend(loc='best', fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # 子圖 2: Reward variance (判斷穩定性)
        axes[1].plot(episodes, moving_std,
                    color=COLORS['secondary'],
                    linewidth=2.0,
                    label='Reward Std Dev')

        # 標註收斂區域
        if moving_std[-1] < moving_std[0] * convergence_threshold:
            axes[1].axhline(moving_std[-1], color=COLORS['success'],
                           linestyle='--', linewidth=1.5, alpha=0.7,
                           label='Converged level')

        axes[1].set_xlabel('Episode', fontsize=10, weight='bold')
        axes[1].set_ylabel('Reward Std Dev', fontsize=10, weight='bold')
        axes[1].set_title('(b) Training Stability', fontsize=11, weight='bold')
        axes[1].legend(loc='best', fontsize=9)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_file, formats=['pdf', 'png'])

    return fig, axes


def main():
    parser = argparse.ArgumentParser(
        description='生成 Learning Curves（標準 RL 論文圖表）'
    )
    parser.add_argument('--data', nargs='+', required=True,
                       help='訓練日誌檔案路徑（可多個）')
    parser.add_argument('--labels', nargs='+', default=None,
                       help='方法標籤（與 --data 對應）')
    parser.add_argument('--output', '-o', type=str,
                       default='figures/learning_curve',
                       help='輸出檔案路徑（不含副檔名）')
    parser.add_argument('--smooth', type=int, default=10,
                       help='平滑窗口大小（預設 10）')
    parser.add_argument('--no-std', action='store_true',
                       help='不顯示標準差區域')
    parser.add_argument('--multi-metric', action='store_true',
                       help='生成多指標圖（Reward + Loss + Handovers）')
    parser.add_argument('--convergence', action='store_true',
                       help='生成收斂性分析圖')

    args = parser.parse_args()

    print("="*70)
    print("Learning Curves 生成器")
    print("="*70)

    # 載入數據
    data_list = []
    for log_file in args.data:
        print(f"\n📖 載入數據: {log_file}")
        data = extract_episode_data(Path(log_file))

        if len(data) == 0:
            print(f"⚠️  警告: {log_file} 無有效數據，跳過")
            continue

        data_list.append(data)

    if len(data_list) == 0:
        print("❌ 錯誤: 無有效數據")
        return 1

    # 設定標籤
    if args.labels:
        if len(args.labels) != len(data_list):
            print(f"⚠️  警告: 標籤數量 ({len(args.labels)}) 與數據數量 ({len(data_list)}) 不符")
            labels = args.labels[:len(data_list)] + \
                     [f'Method {i+1}' for i in range(len(args.labels), len(data_list))]
        else:
            labels = args.labels
    else:
        labels = ['Ours'] if len(data_list) == 1 else \
                 [f'Method {i+1}' for i in range(len(data_list))]

    # 生成學習曲線
    print(f"\n🎨 生成 Learning Curve...")
    plot_learning_curve(data_list, labels, args.output,
                       smooth_window=args.smooth,
                       show_std=not args.no_std)

    # 生成多指標圖（僅當只有一個數據集時）
    if args.multi_metric and len(data_list) == 1:
        print(f"\n🎨 生成多指標圖...")
        multi_output = str(Path(args.output).parent / 'multi_metric_curves')
        plot_multi_metric_curves(data_list[0], multi_output, smooth_window=args.smooth)

    # 生成收斂性分析圖
    if args.convergence and len(data_list) == 1:
        print(f"\n🎨 生成收斂性分析圖...")
        conv_output = str(Path(args.output).parent / 'convergence_analysis')
        plot_convergence_analysis(data_list[0], conv_output)

    print("\n" + "="*70)
    print("✅ Learning Curves 生成完成！")
    print("="*70)

    return 0


if __name__ == '__main__':
    exit(main())
