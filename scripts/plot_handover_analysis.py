#!/usr/bin/env python3
"""
Handover 分析圖表生成器 - 領域特定圖表
展示衛星切換策略的學習過程和合理性

Usage:
    python scripts/plot_handover_analysis.py \\
        --data training_level5_20min_final.log \\
        --output figures/handover_analysis
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# 導入樣式配置
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.paper_style import (
    setup_paper_style, get_figure_size, save_figure,
    COLORS, MARKERS
)
from scripts.extract_training_data import extract_episode_data


def plot_handover_trend(data: pd.DataFrame,
                       output_file: str = 'figures/handover_trend',
                       smooth_window: int = 10):
    """
    繪製 Handover 頻率趨勢圖

    展示：
    1. Agent 如何學習減少不必要的切換
    2. 或者學習增加切換以維持連接品質

    Args:
        data: 訓練數據
        output_file: 輸出檔案路徑
        smooth_window: 平滑窗口大小

    Returns:
        fig, ax: matplotlib Figure 和 Axes 物件
    """

    setup_paper_style('neurips', font_scale=1.0)

    fig, ax = plt.subplots(figsize=get_figure_size())

    episodes = data['episode'].values
    handovers_mean = data['handovers_mean'].values
    handovers_std = data['handovers_std'].values

    # 平滑曲線
    if smooth_window > 1 and len(handovers_mean) >= smooth_window:
        handovers_mean_smooth = uniform_filter1d(handovers_mean, size=smooth_window, mode='nearest')
        handovers_std_smooth = uniform_filter1d(handovers_std, size=smooth_window, mode='nearest')
    else:
        handovers_mean_smooth = handovers_mean
        handovers_std_smooth = handovers_std

    # 繪製主曲線
    ax.plot(episodes, handovers_mean_smooth,
           color=COLORS['tertiary'],
           linewidth=2.5,
           label='Mean Handovers',
           marker=MARKERS[0],
           markersize=4,
           markevery=max(1, len(episodes) // 20))

    # 繪製標準差區域
    ax.fill_between(episodes,
                   handovers_mean_smooth - handovers_std_smooth,
                   handovers_mean_smooth + handovers_std_smooth,
                   color=COLORS['tertiary'],
                   alpha=0.2,
                   label='±σ')

    # 添加參考線（理想範圍）
    # 假設理想的 handover 次數在 10-30 之間
    ax.axhline(20, color=COLORS['info'], linestyle='--',
              linewidth=1.5, alpha=0.5, label='Ideal range')

    ax.set_xlabel('Episode', fontsize=12, weight='bold')
    ax.set_ylabel('Handovers per Episode', fontsize=12, weight='bold')
    ax.set_title('Handover Frequency Trend', fontsize=13, weight='bold')

    ax.legend(loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    save_figure(fig, output_file, formats=['pdf', 'png'])

    return fig, ax


def plot_reward_vs_handovers(data: pd.DataFrame,
                            output_file: str = 'figures/reward_vs_handovers'):
    """
    繪製 Reward vs Handovers 散點圖

    展示：
    - Handover 頻率與 Reward 的關係
    - 是否存在最佳 Handover 頻率

    Args:
        data: 訓練數據
        output_file: 輸出檔案路徑

    Returns:
        fig, ax: matplotlib Figure 和 Axes 物件
    """

    setup_paper_style('neurips', font_scale=1.0)

    fig, ax = plt.subplots(figsize=get_figure_size())

    handovers = data['handovers_mean'].values
    rewards = data['reward_mean'].values
    episodes = data['episode'].values

    # 使用顏色映射表示訓練進度
    scatter = ax.scatter(handovers, rewards,
                        c=episodes,
                        cmap='viridis',
                        s=50,
                        alpha=0.6,
                        edgecolors='white',
                        linewidths=0.5)

    # 添加色條
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Episode', rotation=270, labelpad=20, fontsize=10, weight='bold')

    # 添加趨勢線
    if len(handovers) > 10:
        z = np.polyfit(handovers, rewards, 2)  # 二次多項式擬合
        p = np.poly1d(z)
        x_trend = np.linspace(handovers.min(), handovers.max(), 100)
        ax.plot(x_trend, p(x_trend),
               color=COLORS['danger'],
               linestyle='--',
               linewidth=2.0,
               label='Trend (quadratic fit)',
               alpha=0.7)

    ax.set_xlabel('Handovers per Episode', fontsize=12, weight='bold')
    ax.set_ylabel('Episode Reward', fontsize=12, weight='bold')
    ax.set_title('Reward vs Handover Frequency', fontsize=13, weight='bold')

    ax.legend(loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_file, formats=['pdf', 'png'])

    return fig, ax


def plot_handover_distribution(data: pd.DataFrame,
                              output_file: str = 'figures/handover_distribution'):
    """
    繪製 Handover 分佈圖

    展示：
    - 訓練初期 vs 後期的 Handover 頻率分佈
    - 策略的穩定性

    Args:
        data: 訓練數據
        output_file: 輸出檔案路徑

    Returns:
        fig, ax: matplotlib Figure 和 Axes 物件
    """

    setup_paper_style('neurips', font_scale=1.0)

    fig, ax = plt.subplots(figsize=get_figure_size())

    handovers = data['handovers_mean'].values
    n = len(handovers)

    # 分為三個階段：初期、中期、後期
    stage_size = n // 3
    early = handovers[:stage_size]
    mid = handovers[stage_size:2*stage_size]
    late = handovers[2*stage_size:]

    # 繪製直方圖
    bins = np.linspace(0, handovers.max() * 1.1, 20)

    ax.hist(early, bins=bins, alpha=0.5, color=COLORS['danger'],
           label='Early (First 1/3)', edgecolor='black', linewidth=0.5)

    if len(mid) > 0:
        ax.hist(mid, bins=bins, alpha=0.5, color=COLORS['warning'],
               label='Mid (Middle 1/3)', edgecolor='black', linewidth=0.5)

    if len(late) > 0:
        ax.hist(late, bins=bins, alpha=0.5, color=COLORS['success'],
               label='Late (Last 1/3)', edgecolor='black', linewidth=0.5)

    # 添加均值線
    ax.axvline(early.mean(), color=COLORS['danger'], linestyle='--',
              linewidth=2.0, alpha=0.7)

    if len(late) > 0:
        ax.axvline(late.mean(), color=COLORS['success'], linestyle='--',
                  linewidth=2.0, alpha=0.7)

    ax.set_xlabel('Handovers per Episode', fontsize=12, weight='bold')
    ax.set_ylabel('Frequency', fontsize=12, weight='bold')
    ax.set_title('Handover Distribution: Training Stages', fontsize=13, weight='bold')

    ax.legend(loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_figure(fig, output_file, formats=['pdf', 'png'])

    return fig, ax


def plot_comprehensive_handover_analysis(data: pd.DataFrame,
                                        output_file: str = 'figures/handover_comprehensive',
                                        smooth_window: int = 10):
    """
    綜合 Handover 分析圖（多子圖）

    Args:
        data: 訓練數據
        output_file: 輸出檔案路徑
        smooth_window: 平滑窗口大小

    Returns:
        fig, axes: matplotlib Figure 和 Axes 物件
    """

    setup_paper_style('neurips', font_scale=0.9)

    fig, axes = plt.subplots(2, 2, figsize=get_figure_size(width_ratio=2.0, height_ratio=1.0))

    episodes = data['episode'].values
    handovers_mean = data['handovers_mean'].values
    handovers_std = data['handovers_std'].values
    rewards = data['reward_mean'].values

    # ========================================
    # 子圖 1: Handover 趨勢
    # ========================================
    if smooth_window > 1 and len(handovers_mean) >= smooth_window:
        handovers_smooth = uniform_filter1d(handovers_mean, size=smooth_window, mode='nearest')
        handovers_std_smooth = uniform_filter1d(handovers_std, size=smooth_window, mode='nearest')
    else:
        handovers_smooth = handovers_mean
        handovers_std_smooth = handovers_std

    axes[0, 0].plot(episodes, handovers_smooth,
                   color=COLORS['tertiary'],
                   linewidth=2.0,
                   marker=MARKERS[0],
                   markersize=3,
                   markevery=max(1, len(episodes) // 15))

    axes[0, 0].fill_between(episodes,
                           handovers_smooth - handovers_std_smooth,
                           handovers_smooth + handovers_std_smooth,
                           color=COLORS['tertiary'],
                           alpha=0.2)

    axes[0, 0].set_xlabel('Episode', fontsize=10, weight='bold')
    axes[0, 0].set_ylabel('Handovers', fontsize=10, weight='bold')
    axes[0, 0].set_title('(a) Handover Frequency Trend', fontsize=11, weight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # ========================================
    # 子圖 2: Reward vs Handovers
    # ========================================
    scatter = axes[0, 1].scatter(handovers_mean, rewards,
                                c=episodes,
                                cmap='viridis',
                                s=30,
                                alpha=0.6,
                                edgecolors='white',
                                linewidths=0.3)

    axes[0, 1].set_xlabel('Handovers', fontsize=10, weight='bold')
    axes[0, 1].set_ylabel('Reward', fontsize=10, weight='bold')
    axes[0, 1].set_title('(b) Reward vs Handovers', fontsize=11, weight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # ========================================
    # 子圖 3: Handover 分佈
    # ========================================
    n = len(handovers_mean)
    stage_size = n // 2
    early = handovers_mean[:stage_size]
    late = handovers_mean[stage_size:]

    bins = np.linspace(0, handovers_mean.max() * 1.1, 15)
    axes[1, 0].hist(early, bins=bins, alpha=0.6, color=COLORS['danger'],
                   label='Early', edgecolor='black', linewidth=0.5)
    axes[1, 0].hist(late, bins=bins, alpha=0.6, color=COLORS['success'],
                   label='Late', edgecolor='black', linewidth=0.5)

    axes[1, 0].axvline(early.mean(), color=COLORS['danger'],
                      linestyle='--', linewidth=1.5, alpha=0.7)
    axes[1, 0].axvline(late.mean(), color=COLORS['success'],
                      linestyle='--', linewidth=1.5, alpha=0.7)

    axes[1, 0].set_xlabel('Handovers', fontsize=10, weight='bold')
    axes[1, 0].set_ylabel('Frequency', fontsize=10, weight='bold')
    axes[1, 0].set_title('(c) Handover Distribution', fontsize=11, weight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # ========================================
    # 子圖 4: Handover 變異性
    # ========================================
    axes[1, 1].plot(episodes, handovers_std,
                   color=COLORS['secondary'],
                   linewidth=2.0,
                   marker=MARKERS[1],
                   markersize=3,
                   markevery=max(1, len(episodes) // 15))

    axes[1, 1].set_xlabel('Episode', fontsize=10, weight='bold')
    axes[1, 1].set_ylabel('Handover Std Dev', fontsize=10, weight='bold')
    axes[1, 1].set_title('(d) Strategy Stability', fontsize=11, weight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_file, formats=['pdf', 'png'])

    return fig, axes


def main():
    parser = argparse.ArgumentParser(
        description='生成 Handover 分析圖表（領域特定圖表）'
    )
    parser.add_argument('--data', type=str, required=True,
                       help='訓練日誌檔案路徑')
    parser.add_argument('--output', '-o', type=str,
                       default='figures/handover_analysis',
                       help='輸出檔案路徑（不含副檔名）')
    parser.add_argument('--smooth', type=int, default=10,
                       help='平滑窗口大小（預設 10）')
    parser.add_argument('--comprehensive', action='store_true',
                       help='生成綜合分析圖（2x2 子圖）')

    args = parser.parse_args()

    print("="*70)
    print("Handover 分析圖表生成器")
    print("="*70)

    # 載入數據
    print(f"\n📖 載入數據: {args.data}")
    data = extract_episode_data(Path(args.data))

    if len(data) == 0:
        print("❌ 錯誤: 無有效數據")
        return 1

    # 生成圖表
    if args.comprehensive:
        print(f"\n🎨 生成綜合 Handover 分析圖...")
        plot_comprehensive_handover_analysis(data, args.output, smooth_window=args.smooth)

    else:
        print(f"\n🎨 生成 Handover 趨勢圖...")
        plot_handover_trend(data, args.output, smooth_window=args.smooth)

        print(f"\n🎨 生成 Reward vs Handovers 散點圖...")
        scatter_output = str(Path(args.output).parent / 'reward_vs_handovers')
        plot_reward_vs_handovers(data, scatter_output)

        print(f"\n🎨 生成 Handover 分佈圖...")
        dist_output = str(Path(args.output).parent / 'handover_distribution')
        plot_handover_distribution(data, dist_output)

    print("\n" + "="*70)
    print("✅ Handover 分析圖表生成完成！")
    print("="*70)
    print("\n💡 使用建議:")
    print("   1. 在論文中放置於 Experiments > Domain Analysis 章節")
    print("   2. 展示 Agent 學習到合理的切換策略")
    print("   3. 說明 Handover 頻率與 Reward 的關係")

    return 0


if __name__ == '__main__':
    exit(main())
