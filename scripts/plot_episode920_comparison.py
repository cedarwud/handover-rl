#!/usr/bin/env python3
"""
Episode 920 對比圖生成器 - 核心技術貢獻圖表
證明數值穩定性修復的有效性

這是論文中最重要的圖表，用於展示：
1. 舊版本在 Episode 920 的數值爆炸問題 (loss > 1e6)
2. 新版本的穩定訓練 (loss < 10)

Usage:
    # 生成對比圖 (需要舊版本和新版本的訓練日誌)
    python scripts/plot_episode920_comparison.py \\
        --old training_old_version.log \\
        --new training_level5_20min_final.log \\
        --output figures/episode920_comparison

    # 只畫新版本（如果沒有舊版本數據）
    python scripts/plot_episode920_comparison.py \\
        --new training_level5_20min_final.log \\
        --output figures/episode920_stability
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 導入樣式配置
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.paper_style import (
    setup_paper_style, get_figure_size, save_figure,
    COLORS, MARKERS, LINESTYLES
)
from scripts.extract_training_data import extract_episode_data


def plot_episode920_comparison(old_data: pd.DataFrame = None,
                               new_data: pd.DataFrame = None,
                               output_file: str = 'figures/episode920_comparison',
                               episode_920_focus: bool = True):
    """
    生成 Episode 920 對比圖

    Args:
        old_data: 舊版本數據 (DataFrame)
        new_data: 新版本數據 (DataFrame)
        output_file: 輸出檔案路徑 (不含副檔名)
        episode_920_focus: 是否聚焦 Episode 920 附近

    Returns:
        fig, axes: matplotlib Figure 和 Axes 物件
    """

    # 設定論文樣式
    setup_paper_style('neurips', font_scale=1.1)

    # 決定子圖布局
    if old_data is not None and new_data is not None:
        # 兩個子圖：舊版本 vs 新版本
        fig, axes = plt.subplots(1, 2, figsize=get_figure_size(width_ratio=2.0, height_ratio=0.5))
        ax_old, ax_new = axes
    elif new_data is not None:
        # 只有新版本：單個子圖
        fig, ax_new = plt.subplots(1, 1, figsize=get_figure_size())
        ax_old = None
        axes = [ax_new]
    else:
        raise ValueError("至少需要提供 new_data")

    # ========================================
    # (a) 舊版本 - 展示問題
    # ========================================
    if ax_old is not None and old_data is not None:
        episodes_old = old_data['episode'].values
        loss_old = old_data['loss'].values

        # 繪製 loss 曲線
        ax_old.plot(episodes_old, loss_old,
                   color=COLORS['old_version'],
                   linewidth=2.5,
                   label='Baseline (Unstable)',
                   marker=MARKERS[0],
                   markersize=4,
                   markevery=max(1, len(episodes_old) // 20))

        # 標註 Episode 920 (如果數據中有)
        if 920 in episodes_old:
            idx_920 = np.where(episodes_old == 920)[0][0]
            loss_920 = loss_old[idx_920]

            # 添加垂直線標註
            ax_old.axvline(920, color=COLORS['danger'], linestyle='--',
                          linewidth=1.5, alpha=0.7, label='Episode 920')

            # 添加箭頭標註
            if not np.isnan(loss_920) and not np.isinf(loss_920):
                ax_old.annotate(f'Loss explodes\n({loss_920:.2e})',
                              xy=(920, loss_920),
                              xytext=(920 + 100, loss_920 * 0.5),
                              arrowprops=dict(arrowstyle='->', color=COLORS['danger'],
                                            lw=1.5),
                              fontsize=9,
                              color=COLORS['danger'],
                              weight='bold')

        ax_old.set_xlabel('Episode', fontsize=12, weight='bold')
        ax_old.set_ylabel('Training Loss', fontsize=12, weight='bold')
        ax_old.set_title('(a) Baseline: Numerical Instability', fontsize=13, weight='bold')

        # 使用對數刻度（如果 loss 範圍很大）
        if np.any(loss_old > 100):
            ax_old.set_yscale('log')
            ax_old.set_ylabel('Training Loss (log scale)', fontsize=12, weight='bold')

        ax_old.legend(loc='best', framealpha=0.95)
        ax_old.grid(True, alpha=0.3)

    # ========================================
    # (b) 新版本 - 展示修復
    # ========================================
    if new_data is not None:
        episodes_new = new_data['episode'].values
        loss_new = new_data['loss'].values

        # 繪製 loss 曲線
        ax_new.plot(episodes_new, loss_new,
                   color=COLORS['new_version'],
                   linewidth=2.5,
                   label='Ours (Stable)',
                   marker=MARKERS[1],
                   markersize=4,
                   markevery=max(1, len(episodes_new) // 20))

        # 標註 Episode 920
        if 920 <= episodes_new.max():
            ax_new.axvline(920, color=COLORS['info'], linestyle='--',
                          linewidth=1.5, alpha=0.7, label='Episode 920')

            # 如果 Episode 920 存在於數據中
            if 920 in episodes_new:
                idx_920 = np.where(episodes_new == 920)[0][0]
                loss_920 = loss_new[idx_920]

                if not np.isnan(loss_920) and not np.isinf(loss_920):
                    ax_new.annotate(f'Loss remains stable\n({loss_920:.2f})',
                                  xy=(920, loss_920),
                                  xytext=(920 + 100, loss_920 * 1.5),
                                  arrowprops=dict(arrowstyle='->', color=COLORS['success'],
                                                lw=1.5),
                                  fontsize=9,
                                  color=COLORS['success'],
                                  weight='bold')

        # 添加穩定性區域標註 (loss < 10)
        ax_new.axhline(10, color=COLORS['warning'], linestyle=':',
                      linewidth=1.5, alpha=0.5, label='Stability threshold')

        # 添加陰影區域表示穩定範圍
        ax_new.fill_between(episodes_new, 0, 10,
                           alpha=0.1, color=COLORS['success'],
                           label='Stable region')

        ax_new.set_xlabel('Episode', fontsize=12, weight='bold')
        ax_new.set_ylabel('Training Loss', fontsize=12, weight='bold')

        if ax_old is not None:
            ax_new.set_title('(b) Ours: Numerically Stable', fontsize=13, weight='bold')
        else:
            ax_new.set_title('Training Loss: Numerical Stability', fontsize=13, weight='bold')

        ax_new.legend(loc='best', framealpha=0.95)
        ax_new.grid(True, alpha=0.3)

        # 限制 Y 軸範圍以突出穩定性（如果所有 loss < 100）
        if np.all(loss_new < 100):
            ax_new.set_ylim(bottom=0, top=max(20, loss_new.max() * 1.2))

    plt.tight_layout()

    # 儲存圖表
    save_figure(fig, output_file, formats=['pdf', 'png'])

    return fig, axes


def plot_episode920_zoom(new_data: pd.DataFrame,
                        output_file: str = 'figures/episode920_zoom',
                        window: int = 200):
    """
    Episode 920 附近的放大圖
    詳細展示 Episode 920 前後的訓練穩定性

    Args:
        new_data: 新版本數據
        output_file: 輸出檔案路徑
        window: Episode 920 前後的窗口大小 (預設 ±200)
    """

    setup_paper_style('neurips', font_scale=1.0)

    fig, axes = plt.subplots(2, 1, figsize=get_figure_size(height_ratio=1.0))

    episodes = new_data['episode'].values
    loss = new_data['loss'].values
    reward = new_data['reward_mean'].values

    # 找到 Episode 920 附近的數據
    mask = (episodes >= max(1, 920 - window)) & (episodes <= 920 + window)
    episodes_zoom = episodes[mask]
    loss_zoom = loss[mask]
    reward_zoom = reward[mask]

    # 子圖 1: Loss
    axes[0].plot(episodes_zoom, loss_zoom,
                color=COLORS['new_version'],
                linewidth=2.0,
                marker='o',
                markersize=3)

    axes[0].axvline(920, color=COLORS['danger'], linestyle='--',
                   linewidth=2.0, alpha=0.7, label='Episode 920')

    axes[0].set_ylabel('Training Loss', fontsize=11, weight='bold')
    axes[0].set_title('Episode 920 Zoom-in: Training Loss', fontsize=12, weight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 子圖 2: Reward
    axes[1].plot(episodes_zoom, reward_zoom,
                color=COLORS['primary'],
                linewidth=2.0,
                marker='s',
                markersize=3)

    axes[1].axvline(920, color=COLORS['danger'], linestyle='--',
                   linewidth=2.0, alpha=0.7, label='Episode 920')

    axes[1].set_xlabel('Episode', fontsize=11, weight='bold')
    axes[1].set_ylabel('Episode Reward', fontsize=11, weight='bold')
    axes[1].set_title('Episode 920 Zoom-in: Episode Reward', fontsize=12, weight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_file, formats=['pdf', 'png'])

    return fig, axes


def main():
    parser = argparse.ArgumentParser(
        description='生成 Episode 920 對比圖（論文核心圖表）'
    )
    parser.add_argument('--old', type=str, default=None,
                       help='舊版本訓練日誌（有 Episode 920 bug）')
    parser.add_argument('--new', type=str, required=True,
                       help='新版本訓練日誌（修復後）')
    parser.add_argument('--output', '-o', type=str,
                       default='figures/episode920_comparison',
                       help='輸出檔案路徑（不含副檔名）')
    parser.add_argument('--zoom', action='store_true',
                       help='同時生成 Episode 920 放大圖')
    parser.add_argument('--window', type=int, default=200,
                       help='放大圖窗口大小（預設 ±200 episodes）')

    args = parser.parse_args()

    print("="*70)
    print("Episode 920 對比圖生成器")
    print("="*70)

    # 載入數據
    old_data = None
    if args.old:
        print(f"\n📖 載入舊版本數據: {args.old}")
        old_data = extract_episode_data(Path(args.old))

    print(f"\n📖 載入新版本數據: {args.new}")
    new_data = extract_episode_data(Path(args.new))

    if len(new_data) == 0:
        print("❌ 錯誤: 無法從新版本日誌提取數據")
        return 1

    # 生成主對比圖
    print(f"\n🎨 生成 Episode 920 對比圖...")
    fig, axes = plot_episode920_comparison(old_data, new_data, args.output)

    # 生成放大圖
    if args.zoom and 920 in new_data['episode'].values:
        print(f"\n🔍 生成 Episode 920 放大圖...")
        zoom_output = str(Path(args.output).parent / 'episode920_zoom')
        plot_episode920_zoom(new_data, zoom_output, window=args.window)

    elif args.zoom:
        print(f"\n⚠️  警告: 訓練尚未到達 Episode 920，無法生成放大圖")
        print(f"    當前最大 Episode: {new_data['episode'].max()}")

    print("\n" + "="*70)
    print("✅ Episode 920 圖表生成完成！")
    print("="*70)
    print(f"\n💡 使用建議:")
    print(f"   1. 在論文中放置於 Experiments > Numerical Stability 章節")
    print(f"   2. Caption 建議:")
    print(f"      'Training loss comparison at Episode 920. (a) Baseline")
    print(f"       suffers numerical explosion. (b) Our method maintains")
    print(f"       stability with 4-layer numerical enhancement.'")
    print(f"   3. 強調這是技術貢獻的核心證明")

    return 0


if __name__ == '__main__':
    exit(main())
