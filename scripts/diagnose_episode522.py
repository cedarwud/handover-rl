#!/usr/bin/env python3
"""
Diagnostic Script: Episode 522 問題診斷

目的: 精確定位 2025-10-13 14:40-16:30 時間範圍的問題根源

測試方案:
1. 單獨測試 Episode 522 (10分鐘窗口，20分鐘 duration)
2. 檢查該時間範圍的軌道數據特徵
3. 測量環境處理時間
"""

import sys
import time
from datetime import datetime, timedelta
import logging
import numpy as np

# 設置路徑
sys.path.insert(0, '/home/sat/satellite/handover-rl/src')

from environments.satellite_handover_env import SatelliteHandoverEnv
from adapters.orbit_precompute_table import OrbitPrecomputeTable
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """載入配置"""
    with open('config/diagnostic_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def diagnose_episode_522():
    """診斷 Episode 522"""

    logger.info("=" * 80)
    logger.info("Episode 522 診斷測試開始")
    logger.info("=" * 80)

    # 載入配置
    config = load_config()

    # Episode 522 參數
    start_time = datetime(2025, 10, 13, 15, 0, 0)  # Episode 522 起始時間
    episode_duration_minutes = 20
    time_step_seconds = 5

    logger.info(f"測試時間範圍: {start_time} ~ {start_time + timedelta(minutes=episode_duration_minutes)}")
    logger.info(f"Episode duration: {episode_duration_minutes} minutes")
    logger.info(f"Time step: {time_step_seconds} seconds")

    # 創建環境
    logger.info("\n" + "=" * 80)
    logger.info("步驟 1: 創建環境")
    logger.info("=" * 80)

    try:
        env = SatelliteHandoverEnv(config=config)
        logger.info("✅ 環境創建成功")
    except Exception as e:
        logger.error(f"❌ 環境創建失敗: {e}")
        return

    # 測試 reset
    logger.info("\n" + "=" * 80)
    logger.info("步驟 2: 測試 env.reset()")
    logger.info("=" * 80)

    reset_start = time.time()
    try:
        obs, info = env.reset(options={'episode_start_time': start_time})
        reset_time = time.time() - reset_start
        logger.info(f"✅ Reset 成功，耗時: {reset_time:.2f}s")
        logger.info(f"   觀測空間 shape: {obs.shape}")
        logger.info(f"   可見衛星數量: {info.get('num_visible_satellites', 'N/A')}")
    except Exception as e:
        logger.error(f"❌ Reset 失敗: {e}")
        import traceback
        traceback.print_exc()
        return

    # 測試 step
    logger.info("\n" + "=" * 80)
    logger.info("步驟 3: 測試 10 個 steps")
    logger.info("=" * 80)

    step_times = []
    visible_satellites_history = []

    for step_idx in range(10):
        step_start = time.time()

        try:
            # 隨機動作
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            step_time = time.time() - step_start
            step_times.append(step_time)

            num_visible = info.get('num_visible_satellites', 0)
            visible_satellites_history.append(num_visible)

            logger.info(f"  Step {step_idx}: {step_time:.3f}s, visible={num_visible}, reward={reward:.2f}")

            if step_time > 1.0:
                logger.warning(f"  ⚠️  Step {step_idx} 處理時間過長: {step_time:.3f}s")

        except Exception as e:
            logger.error(f"❌ Step {step_idx} 失敗: {e}")
            import traceback
            traceback.print_exc()
            break

    # 統計分析
    logger.info("\n" + "=" * 80)
    logger.info("步驟 4: 統計分析")
    logger.info("=" * 80)

    if step_times:
        logger.info(f"Step 時間統計:")
        logger.info(f"  平均: {np.mean(step_times):.3f}s")
        logger.info(f"  最小: {np.min(step_times):.3f}s")
        logger.info(f"  最大: {np.max(step_times):.3f}s")
        logger.info(f"  標準差: {np.std(step_times):.3f}s")

    if visible_satellites_history:
        logger.info(f"\n可見衛星統計:")
        logger.info(f"  平均: {np.mean(visible_satellites_history):.1f}")
        logger.info(f"  最小: {np.min(visible_satellites_history)}")
        logger.info(f"  最大: {np.max(visible_satellites_history)}")

    # 檢查 HDF5 數據
    logger.info("\n" + "=" * 80)
    logger.info("步驟 5: 檢查 HDF5 數據")
    logger.info("=" * 80)

    try:
        # 獲取 adapter
        adapter = env.orbit_adapter

        if hasattr(adapter, 'table'):
            table = adapter.table
            logger.info(f"✅ 使用 Precompute Table")
            logger.info(f"   HDF5 路徑: {table.hdf5_path}")

            # 檢查該時間範圍的數據
            test_times = [
                start_time,
                start_time + timedelta(minutes=5),
                start_time + timedelta(minutes=10),
                start_time + timedelta(minutes=15),
            ]

            satellite_ids = table.satellite_ids[:5]  # 測試前5個衛星

            logger.info(f"\n測試衛星 (前5個): {satellite_ids}")

            for test_time in test_times:
                logger.info(f"\n時間: {test_time}")

                query_start = time.time()

                for sat_id in satellite_ids:
                    try:
                        state = table.get_state(sat_id, test_time)
                        rsrp = state.get('rsrp_dbm', np.nan)
                        logger.info(f"  {sat_id}: RSRP={rsrp:.2f} dBm")
                    except Exception as e:
                        logger.error(f"  {sat_id}: 查詢失敗 - {e}")

                query_time = time.time() - query_start
                logger.info(f"  查詢時間: {query_time:.3f}s")

                if query_time > 0.5:
                    logger.warning(f"  ⚠️  查詢時間過長!")

    except Exception as e:
        logger.error(f"❌ HDF5 檢查失敗: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\n" + "=" * 80)
    logger.info("診斷完成")
    logger.info("=" * 80)

def compare_time_ranges():
    """比較不同時間範圍的性能"""

    logger.info("\n" + "=" * 80)
    logger.info("比較測試: 正常時間 vs 問題時間")
    logger.info("=" * 80)

    config = load_config()

    # 測試時間範圍
    test_ranges = [
        ("正常時間 (10-11 00:00)", datetime(2025, 10, 11, 0, 0, 0)),
        ("問題時間 (10-13 15:00)", datetime(2025, 10, 13, 15, 0, 0)),
        ("問題時間 (10-13 16:00)", datetime(2025, 10, 13, 16, 0, 0)),
    ]

    results = []

    for name, start_time in test_ranges:
        logger.info(f"\n測試: {name}")
        logger.info(f"時間: {start_time}")

        try:
            env = SatelliteHandoverEnv(config=config)

            # Reset
            reset_start = time.time()
            obs, info = env.reset(options={'episode_start_time': start_time})
            reset_time = time.time() - reset_start

            # 10 steps
            step_times = []
            for _ in range(10):
                step_start = time.time()
                action = env.action_space.sample()
                env.step(action)
                step_times.append(time.time() - step_start)

            avg_step_time = np.mean(step_times)

            logger.info(f"  Reset 時間: {reset_time:.3f}s")
            logger.info(f"  平均 Step 時間: {avg_step_time:.3f}s")
            logger.info(f"  可見衛星: {info.get('num_visible_satellites', 'N/A')}")

            results.append({
                'name': name,
                'reset_time': reset_time,
                'avg_step_time': avg_step_time,
                'visible': info.get('num_visible_satellites', 0)
            })

        except Exception as e:
            logger.error(f"  ❌ 測試失敗: {e}")

    # 比較結果
    if len(results) > 1:
        logger.info("\n" + "=" * 80)
        logger.info("比較結果")
        logger.info("=" * 80)

        baseline = results[0]
        for result in results[1:]:
            logger.info(f"\n{result['name']} vs {baseline['name']}:")
            logger.info(f"  Reset 時間: {result['reset_time']:.3f}s vs {baseline['reset_time']:.3f}s "
                       f"({result['reset_time']/baseline['reset_time']:.1f}x)")
            logger.info(f"  Step 時間: {result['avg_step_time']:.3f}s vs {baseline['avg_step_time']:.3f}s "
                       f"({result['avg_step_time']/baseline['avg_step_time']:.1f}x)")
            logger.info(f"  可見衛星: {result['visible']} vs {baseline['visible']}")

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("Episode 522 問題診斷工具")
    print("=" * 80)
    print()

    # 診斷 Episode 522
    diagnose_episode_522()

    # 比較不同時間範圍
    compare_time_ranges()

    print("\n診斷完成!")
