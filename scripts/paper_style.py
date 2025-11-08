#!/usr/bin/env python3
"""
論文級圖表樣式配置
符合 IEEE / NeurIPS / ICML / ICLR 標準

Usage:
    from scripts.paper_style import setup_paper_style, COLORS, MARKERS

    setup_paper_style()  # 或 setup_paper_style('ieee') / setup_paper_style('neurips')

    plt.plot(x, y, color=COLORS['primary'], marker=MARKERS[0])
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from typing import Optional


# ============================================================================
# 色盲友好配色方案 (符合學術標準)
# ============================================================================

# 主要配色 (基於 ColorBrewer 色盲友好方案)
COLORS = {
    # 主色調
    'primary': '#1f77b4',      # 藍色 - 我們的方法
    'secondary': '#ff7f0e',    # 橙色 - Baseline
    'tertiary': '#2ca02c',     # 綠色 - 第三個方法
    'quaternary': '#d62728',   # 紅色 - 強調/錯誤

    # 用於對比
    'success': '#2ca02c',      # 綠色 - 成功/改進
    'danger': '#d62728',       # 紅色 - 失敗/問題
    'warning': '#ff7f0e',      # 橙色 - 警告
    'info': '#1f77b4',         # 藍色 - 信息

    # 灰階
    'gray_dark': '#2f2f2f',
    'gray_medium': '#7f7f7f',
    'gray_light': '#cfcfcf',

    # Episode 920 對比專用
    'old_version': '#d62728',  # 紅色 - 舊版本（有問題）
    'new_version': '#2ca02c',  # 綠色 - 新版本（修復後）
}

# 色盲友好的多色系列（用於多條曲線對比）
COLOR_PALETTE = [
    '#1f77b4',  # 藍色
    '#ff7f0e',  # 橙色
    '#2ca02c',  # 綠色
    '#d62728',  # 紅色
    '#9467bd',  # 紫色
    '#8c564b',  # 棕色
    '#e377c2',  # 粉色
    '#7f7f7f',  # 灰色
]

# 標記樣式
MARKERS = ['o', 's', '^', 'v', 'D', 'P', '*', 'X']

# 線條樣式
LINESTYLES = ['-', '--', '-.', ':']


# ============================================================================
# 樣式設定函數
# ============================================================================

def setup_paper_style(style: str = 'default', font_scale: float = 1.0):
    """
    設定論文級圖表樣式

    Args:
        style: 樣式名稱
            - 'default': 通用學術樣式 (推薦)
            - 'ieee': IEEE 期刊/會議樣式
            - 'neurips': NeurIPS/ICML/ICLR 樣式
            - 'nature': Nature 期刊樣式 (需要字體)
        font_scale: 字體縮放因子 (預設 1.0)

    Returns:
        None (直接修改 matplotlib 全局配置)
    """

    # 重置為預設設定
    mpl.rcParams.update(mpl.rcParamsDefault)

    # ========================================
    # 通用設定 (所有樣式共享)
    # ========================================

    base_fontsize = 10 * font_scale

    plt.rcParams.update({
        # 圖片品質
        'figure.dpi': 100,              # 螢幕顯示 DPI
        'savefig.dpi': 300,             # 儲存 DPI (印刷品質)
        'savefig.format': 'pdf',        # 預設儲存格式 (vector)
        'savefig.bbox': 'tight',        # 自動裁切空白
        'savefig.pad_inches': 0.05,     # 邊距

        # 字型設定
        'font.family': 'serif',         # 字型家族
        'font.size': base_fontsize,     # 基礎字型大小
        'axes.labelsize': base_fontsize,      # 軸標籤字型大小
        'axes.titlesize': base_fontsize + 1,  # 子圖標題字型大小
        'xtick.labelsize': base_fontsize - 1, # X軸刻度標籤
        'ytick.labelsize': base_fontsize - 1, # Y軸刻度標籤
        'legend.fontsize': base_fontsize - 1, # 圖例字型大小

        # 線條與標記
        'lines.linewidth': 2.0,         # 線條寬度
        'lines.markersize': 6,          # 標記大小
        'lines.markeredgewidth': 0.5,   # 標記邊框寬度

        # 軸設定
        'axes.linewidth': 1.0,          # 軸線寬度
        'axes.grid': True,              # 預設顯示網格
        'axes.axisbelow': True,         # 網格在圖形下方
        'axes.edgecolor': '#2f2f2f',    # 軸邊框顏色
        'axes.labelcolor': '#2f2f2f',   # 軸標籤顏色

        # 網格設定
        'grid.alpha': 0.3,              # 網格透明度
        'grid.linestyle': '--',         # 網格線樣式
        'grid.linewidth': 0.5,          # 網格線寬度

        # 圖例設定
        'legend.frameon': True,         # 圖例框架
        'legend.framealpha': 0.9,       # 圖例框架透明度
        'legend.fancybox': True,        # 圓角框架
        'legend.edgecolor': '#cfcfcf',  # 框架邊框顏色

        # 刻度設定
        'xtick.direction': 'in',        # 刻度方向 (向內)
        'ytick.direction': 'in',        # 刻度方向 (向內)
        'xtick.major.size': 4,          # 主刻度長度
        'ytick.major.size': 4,          # 主刻度長度
        'xtick.minor.size': 2,          # 次刻度長度
        'ytick.minor.size': 2,          # 次刻度長度

        # 顏色循環 (使用我們的色盲友好配色)
        'axes.prop_cycle': plt.cycler(color=COLOR_PALETTE),
    })

    # ========================================
    # 樣式特定設定
    # ========================================

    if style == 'ieee':
        # IEEE 樣式 (Times New Roman, 較小字型)
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 8 * font_scale,
            'axes.labelsize': 8 * font_scale,
            'axes.titlesize': 9 * font_scale,
            'xtick.labelsize': 7 * font_scale,
            'ytick.labelsize': 7 * font_scale,
            'legend.fontsize': 7 * font_scale,
            'lines.linewidth': 1.5,
        })

    elif style == 'neurips':
        # NeurIPS/ICML/ICLR 樣式 (較大字型, 清晰)
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['DejaVu Sans', 'Arial'],
            'font.size': 11 * font_scale,
            'axes.labelsize': 12 * font_scale,
            'axes.titlesize': 13 * font_scale,
            'lines.linewidth': 2.5,
        })

    elif style == 'nature':
        # Nature 樣式 (Helvetica/Arial, 精緻)
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Helvetica', 'Arial'],
            'font.size': 8 * font_scale,
            'axes.labelsize': 9 * font_scale,
            'lines.linewidth': 1.0,
            'axes.linewidth': 0.75,
        })

    # 設定 seaborn 默認樣式
    sns.set_palette(COLOR_PALETTE)

    print(f"✅ 論文樣式已設定: {style} (font_scale={font_scale})")


def get_figure_size(width_ratio: float = 1.0,
                    height_ratio: float = 0.618,
                    base_width: float = 6.0) -> tuple:
    """
    計算圖表尺寸 (遵循黃金比例)

    Args:
        width_ratio: 寬度比例 (相對於 base_width)
        height_ratio: 高度比例 (相對於寬度, 預設黃金比例 0.618)
        base_width: 基礎寬度 (inches)

    Returns:
        (width, height) in inches

    Examples:
        # 單欄圖 (標準)
        fig, ax = plt.subplots(figsize=get_figure_size())

        # 雙欄圖 (寬度 2 倍)
        fig, ax = plt.subplots(figsize=get_figure_size(width_ratio=2.0))

        # 方形圖
        fig, ax = plt.subplots(figsize=get_figure_size(height_ratio=1.0))
    """
    width = base_width * width_ratio
    height = width * height_ratio
    return (width, height)


def save_figure(fig, filename: str, formats: list = ['pdf', 'png'], **kwargs):
    """
    儲存圖表 (多種格式)

    Args:
        fig: matplotlib Figure 物件
        filename: 檔案名稱 (不含副檔名)
        formats: 儲存格式列表
        **kwargs: 傳遞給 savefig 的參數

    Examples:
        save_figure(fig, 'figures/learning_curve')
        save_figure(fig, 'figures/episode920', formats=['pdf', 'png', 'svg'])
    """
    from pathlib import Path

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        output_file = filename.with_suffix(f'.{fmt}')
        fig.savefig(output_file, format=fmt, **kwargs)
        print(f"💾 圖表已儲存: {output_file}")


# ============================================================================
# 預設設定
# ============================================================================

def reset_style():
    """重置為 matplotlib 預設樣式"""
    mpl.rcParams.update(mpl.rcParamsDefault)
    sns.reset_defaults()
    print("✅ 樣式已重置為預設")


# ============================================================================
# 使用範例
# ============================================================================

if __name__ == '__main__':
    import numpy as np

    # 設定論文樣式
    setup_paper_style('default')

    # 創建測試圖表
    x = np.linspace(0, 10, 100)

    fig, axes = plt.subplots(2, 2, figsize=get_figure_size(width_ratio=2.0, height_ratio=1.0))

    # 測試不同配色
    axes[0, 0].plot(x, np.sin(x), color=COLORS['primary'], label='Primary')
    axes[0, 0].plot(x, np.cos(x), color=COLORS['secondary'], label='Secondary')
    axes[0, 0].set_title('Color Test')
    axes[0, 0].legend()

    # 測試標記樣式
    for i, (marker, linestyle) in enumerate(zip(MARKERS[:3], LINESTYLES[:3])):
        axes[0, 1].plot(x[::10], np.sin(x[::10] + i),
                        marker=marker, linestyle=linestyle,
                        label=f'Style {i+1}')
    axes[0, 1].set_title('Marker & Line Test')
    axes[0, 1].legend()

    # 測試誤差帶
    y = np.sin(x)
    y_err = 0.2
    axes[1, 0].plot(x, y, color=COLORS['primary'], label='Mean')
    axes[1, 0].fill_between(x, y - y_err, y + y_err,
                             alpha=0.3, color=COLORS['primary'],
                             label='±σ')
    axes[1, 0].set_title('Error Band Test')
    axes[1, 0].legend()

    # 測試 Episode 920 配色
    axes[1, 1].plot(x, x**2, color=COLORS['old_version'],
                     linewidth=3, label='Old Version')
    axes[1, 1].plot(x, x*10, color=COLORS['new_version'],
                     linewidth=3, label='New Version')
    axes[1, 1].set_title('Episode 920 Colors')
    axes[1, 1].legend()

    plt.tight_layout()

    # 儲存測試圖表
    save_figure(fig, 'test_paper_style', formats=['png'])

    plt.show()

    print("\n✅ 樣式測試完成！")
    print("   查看 test_paper_style.png")
