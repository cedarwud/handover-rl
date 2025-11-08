#!/usr/bin/env python3
"""
性能對比表格生成器
生成論文級的性能對比表格（LaTeX / Markdown）

Usage:
    # 生成 LaTeX 表格
    python scripts/generate_performance_table.py \\
        --data ours.log baseline1.log baseline2.log \\
        --labels "Ours" "Baseline 1" "Baseline 2" \\
        --output tables/performance_comparison.tex

    # 生成 Markdown 表格（用於 README）
    python scripts/generate_performance_table.py \\
        --data ours.log baseline1.log \\
        --labels "Ours" "Baseline" \\
        --format markdown \\
        --output tables/performance_comparison.md
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 導入數據提取工具
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.extract_training_data import extract_episode_data, compute_statistics


def generate_latex_table(results: pd.DataFrame,
                         output_file: str = None,
                         caption: str = None,
                         label: str = 'tab:performance') -> str:
    """
    生成 LaTeX 表格

    Args:
        results: 性能結果 DataFrame
        output_file: 輸出檔案路徑
        caption: 表格標題
        label: LaTeX 標籤

    Returns:
        LaTeX 表格字符串
    """

    if caption is None:
        caption = "Performance comparison of different methods on LEO satellite handover task."

    # LaTeX 表格模板
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"    \centering")
    latex.append(r"    \caption{" + caption + r"}")
    latex.append(r"    \label{" + label + r"}")

    # 表格列數
    n_cols = len(results.columns)

    # 表格對齊（第一列左對齊，其他居中）
    col_align = "l" + "c" * (n_cols - 1)

    latex.append(r"    \begin{tabular}{" + col_align + r"}")
    latex.append(r"        \toprule")

    # 表頭
    headers = " & ".join(results.columns) + r" \\"
    latex.append(r"        " + headers)
    latex.append(r"        \midrule")

    # 數據行
    for _, row in results.iterrows():
        # 格式化數值
        formatted_row = []
        for i, (col, val) in enumerate(zip(results.columns, row)):
            if i == 0:  # 方法名稱
                formatted_row.append(str(val))
            elif isinstance(val, str):
                formatted_row.append(val)
            else:
                # 數值：保留 2 位小數
                formatted_row.append(f"{val:.2f}" if not np.isnan(val) else "-")

        row_str = " & ".join(formatted_row) + r" \\"
        latex.append(r"        " + row_str)

    latex.append(r"        \bottomrule")
    latex.append(r"    \end{tabular}")
    latex.append(r"\end{table}")

    latex_str = "\n".join(latex)

    # 儲存到檔案
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(latex_str)

        print(f"💾 LaTeX 表格已儲存: {output_path}")

    return latex_str


def generate_markdown_table(results: pd.DataFrame,
                           output_file: str = None) -> str:
    """
    生成 Markdown 表格

    Args:
        results: 性能結果 DataFrame
        output_file: 輸出檔案路徑

    Returns:
        Markdown 表格字符串
    """

    # 使用 pandas 的 to_markdown() 方法
    markdown = results.to_markdown(index=False, floatfmt=".2f")

    # 儲存到檔案
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(markdown)

        print(f"💾 Markdown 表格已儲存: {output_path}")

    return markdown


def create_performance_comparison(data_files: list,
                                  labels: list,
                                  include_steps: bool = True) -> pd.DataFrame:
    """
    創建性能對比表格

    Args:
        data_files: 訓練日誌檔案列表
        labels: 方法標籤列表
        include_steps: 是否包含訓練步數

    Returns:
        性能對比 DataFrame
    """

    results = {
        'Method': [],
        'Final Reward': [],
        'Best Reward': [],
        'Avg Handovers': [],
        'Final Loss': [],
    }

    if include_steps:
        results['Training Episodes'] = []

    for log_file, label in zip(data_files, labels):
        print(f"📊 分析: {label} ({log_file})")

        # 提取數據
        data = extract_episode_data(Path(log_file))

        if len(data) == 0:
            print(f"⚠️  警告: {log_file} 無有效數據")
            continue

        # 計算統計量
        stats = compute_statistics(data)

        # 添加到結果
        results['Method'].append(label)

        # Final Reward (mean ± std)
        if stats['final_reward_mean'] is not None:
            results['Final Reward'].append(
                f"{stats['final_reward_mean']:.2f}±{stats['final_reward_std']:.2f}"
            )
        else:
            results['Final Reward'].append("-")

        # Best Reward
        if stats['best_reward'] is not None:
            results['Best Reward'].append(stats['best_reward'])
        else:
            results['Best Reward'].append(np.nan)

        # Avg Handovers
        if stats['avg_handovers'] is not None:
            results['Avg Handovers'].append(stats['avg_handovers'])
        else:
            results['Avg Handovers'].append(np.nan)

        # Final Loss
        if stats['final_loss'] is not None:
            results['Final Loss'].append(stats['final_loss'])
        else:
            results['Final Loss'].append(np.nan)

        # Training Episodes
        if include_steps:
            results['Training Episodes'].append(stats['total_episodes'])

    return pd.DataFrame(results)


def create_ablation_study_table(data_files: list,
                                labels: list,
                                baseline_idx: int = 0) -> pd.DataFrame:
    """
    創建 Ablation Study 表格（顯示相對改進）

    Args:
        data_files: 訓練日誌檔案列表
        labels: 方法標籤列表
        baseline_idx: Baseline 方法的索引

    Returns:
        Ablation Study DataFrame
    """

    # 先創建基本的性能對比表格
    basic_results = create_performance_comparison(data_files, labels, include_steps=False)

    # 提取 Baseline 的性能
    baseline_reward_str = basic_results.iloc[baseline_idx]['Final Reward']
    baseline_reward = float(baseline_reward_str.split('±')[0])

    # 添加相對改進列
    improvements = []
    for _, row in basic_results.iterrows():
        reward_str = row['Final Reward']
        if reward_str == "-":
            improvements.append("-")
        else:
            reward = float(reward_str.split('±')[0])
            improvement = ((reward - baseline_reward) / abs(baseline_reward)) * 100
            improvements.append(f"{improvement:+.1f}%")

    basic_results['Improvement'] = improvements

    return basic_results


def main():
    parser = argparse.ArgumentParser(
        description='生成性能對比表格（LaTeX / Markdown）'
    )
    parser.add_argument('--data', nargs='+', required=True,
                       help='訓練日誌檔案路徑（可多個）')
    parser.add_argument('--labels', nargs='+', required=True,
                       help='方法標籤（與 --data 對應）')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='輸出檔案路徑')
    parser.add_argument('--format', choices=['latex', 'markdown'], default='latex',
                       help='輸出格式（預設 latex）')
    parser.add_argument('--caption', type=str, default=None,
                       help='表格標題（僅 LaTeX）')
    parser.add_argument('--label', type=str, default='tab:performance',
                       help='LaTeX 標籤（預設 tab:performance）')
    parser.add_argument('--ablation', action='store_true',
                       help='生成 Ablation Study 表格（顯示相對改進）')
    parser.add_argument('--baseline-idx', type=int, default=0,
                       help='Baseline 方法的索引（預設 0）')

    args = parser.parse_args()

    # 檢查數據和標籤數量是否匹配
    if len(args.data) != len(args.labels):
        print(f"❌ 錯誤: 數據檔案數量 ({len(args.data)}) 與標籤數量 ({len(args.labels)}) 不符")
        return 1

    print("="*70)
    print("性能對比表格生成器")
    print("="*70)

    # 創建性能對比表格
    if args.ablation:
        print("\n📊 生成 Ablation Study 表格...")
        results = create_ablation_study_table(args.data, args.labels, args.baseline_idx)
    else:
        print("\n📊 生成性能對比表格...")
        results = create_performance_comparison(args.data, args.labels)

    # 顯示表格預覽
    print("\n" + "="*70)
    print("📋 表格預覽:")
    print("="*70)
    print(results.to_string(index=False))
    print("="*70)

    # 生成輸出
    if args.format == 'latex':
        latex_str = generate_latex_table(results, args.output, args.caption, args.label)
        print("\n" + "="*70)
        print("📄 LaTeX 程式碼:")
        print("="*70)
        print(latex_str)
        print("="*70)
        print("\n💡 使用建議:")
        print("   1. 複製上述 LaTeX 程式碼到論文中")
        print("   2. 確保 preamble 中有: \\usepackage{booktabs}")
        print("   3. 表格會自動置頂 (table[t])")

    else:  # markdown
        md_str = generate_markdown_table(results, args.output)
        print("\n" + "="*70)
        print("📄 Markdown 程式碼:")
        print("="*70)
        print(md_str)
        print("="*70)

    print("\n✅ 表格生成完成！")

    return 0


if __name__ == '__main__':
    exit(main())
