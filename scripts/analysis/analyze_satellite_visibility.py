#!/usr/bin/env python3
"""
Satellite Visibility Analysis (Independent)

獨立於訓練之外，分析實際有多少衛星在指定時間範圍內可見

Academic Compliance:
- ZERO HARDCODING: 結果完全由實際軌道數據決定
- DATA-DRIVEN: 基於真實TLE + 地面站位置 + 時間範圍
- TRANSPARENT: 顯示完整分析過程
"""

import sys
import yaml
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from adapters.orbit_engine_adapter import OrbitEngineAdapter
from utils.satellite_utils import extract_satellites_from_tle, get_default_tle_path


def analyze_visibility(
    adapter,
    satellite_ids,
    start_time,
    end_time,
    time_step_minutes=60,
    min_elevation=10.0
):
    """
    分析衛星可見性

    Returns:
        dict: {
            'visible_satellites': list of sat_ids,
            'visibility_count': dict of {sat_id: count},
            'total_time_points': int
        }
    """
    print(f"\n🔍 開始可見性分析...")
    print(f"   時間範圍: {start_time} 到 {end_time}")
    print(f"   時間步長: {time_step_minutes} 分鐘")
    print(f"   最低仰角: {min_elevation}°")
    print(f"   候選衛星: {len(satellite_ids)} 顆")

    visible_satellites = set()
    visibility_count = {}

    current_time = start_time
    time_points = 0

    print(f"\n⏳ 正在查詢... (每個時間點查詢 {len(satellite_ids)} 顆衛星)")

    while current_time <= end_time:
        if time_points % 6 == 0:  # 每6小時顯示進度
            print(f"   進度: {current_time.strftime('%Y-%m-%d %H:%M')}")

        for sat_id in satellite_ids:
            try:
                state = adapter.calculate_state(sat_id, current_time)

                if state and state.get('elevation', -90) >= min_elevation:
                    visible_satellites.add(sat_id)
                    visibility_count[sat_id] = visibility_count.get(sat_id, 0) + 1

            except Exception as e:
                # 衛星可能沒有有效的TLE數據
                pass

        time_points += 1
        current_time += timedelta(minutes=time_step_minutes)

    return {
        'visible_satellites': sorted(list(visible_satellites)),
        'visibility_count': visibility_count,
        'total_time_points': time_points
    }


def main():
    print("=" * 80)
    print("🛰️  衛星可見性分析 (獨立分析，不進行訓練)")
    print("=" * 80)
    print("目的: 確定實際有多少顆衛星在訓練時間範圍內可見")
    print("方法: 實際查詢每顆候選衛星的軌道位置")
    print("=" * 80)

    # 載入配置
    config_path = Path(__file__).parent / "config" / "data_gen_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"\n✅ 配置已載入")
    print(f"   地面站: ({config['ground_station']['latitude']}°N, "
          f"{config['ground_station']['longitude']}°E)")

    # 初始化adapter
    print(f"\n🔧 初始化OrbitEngineAdapter...")
    adapter = OrbitEngineAdapter(config)
    print(f"✅ Adapter已初始化")

    # 提取候選衛星
    print(f"\n📡 從TLE提取候選衛星...")
    tle_path = get_default_tle_path()

    # 測試不同數量的候選衛星
    candidate_counts = [200, 500]  # 先測試200和500

    for max_candidates in candidate_counts:
        print(f"\n{'='*80}")
        print(f"測試案例: 前 {max_candidates} 顆候選衛星")
        print(f"{'='*80}")

        candidates = extract_satellites_from_tle(tle_path, max_satellites=max_candidates)
        print(f"✅ 已提取 {len(candidates)} 顆候選衛星")
        print(f"   範圍: {candidates[0]} 到 {candidates[-1]}")

        # 分析24小時可見性（代表性樣本）
        start_time = datetime(2025, 10, 7, 0, 0, 0)
        end_time = datetime(2025, 10, 8, 0, 0, 0)

        results = analyze_visibility(
            adapter=adapter,
            satellite_ids=candidates,
            start_time=start_time,
            end_time=end_time,
            time_step_minutes=60,  # 每小時採樣
            min_elevation=10.0
        )

        visible_sats = results['visible_satellites']
        visibility_count = results['visibility_count']

        print(f"\n📊 分析結果:")
        print(f"   候選衛星: {len(candidates)} 顆")
        print(f"   可見衛星: {len(visible_sats)} 顆")
        print(f"   可見率: {len(visible_sats)/len(candidates)*100:.1f}%")
        print(f"   時間點數: {results['total_time_points']}")

        # 顯示可見性統計
        if visibility_count:
            counts = list(visibility_count.values())
            print(f"\n   可見性統計:")
            print(f"      平均可見次數: {sum(counts)/len(counts):.1f}")
            print(f"      最多可見次數: {max(counts)}")
            print(f"      最少可見次數: {min(counts)}")

        # 顯示前10和後10顆可見衛星
        if len(visible_sats) > 0:
            print(f"\n   前10顆可見衛星: {visible_sats[:10]}")
            print(f"   後10顆可見衛星: {visible_sats[-10:]}")

    # 最終建議
    print(f"\n{'='*80}")
    print("🎯 建議的衛星池大小")
    print(f"{'='*80}")
    print(f"基於24小時可見性分析，建議使用:")
    if len(visible_sats) > 0:
        print(f"   衛星池大小: {len(visible_sats)} 顆")
        print(f"   數據來源: 實際軌道計算（零硬編碼）")
        print(f"   可重現性: 相同TLE + 時間範圍 → 相同結果")

    print(f"\n與原始計劃(125顆)比較:")
    if len(visible_sats) > 0:
        diff = len(visible_sats) - 125
        if diff > 0:
            print(f"   實際可見: {len(visible_sats)} 顆 (比125多 {diff} 顆)")
        elif diff < 0:
            print(f"   實際可見: {len(visible_sats)} 顆 (比125少 {-diff} 顆)")
        else:
            print(f"   實際可見: {len(visible_sats)} 顆 (剛好125顆！)")

    print(f"\n{'='*80}")
    print("✅ 可見性分析完成")
    print(f"{'='*80}")
    print("\n下一步: 使用這個結果訓練，或選擇其他pool size")


if __name__ == "__main__":
    main()
