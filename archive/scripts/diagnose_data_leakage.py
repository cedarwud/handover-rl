#!/usr/bin/env python3
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

from adapters.handover_event_loader import create_handover_event_loader

print("=" * 80)
print("診斷數據洩漏根本原因")
print("=" * 80)

loader = create_handover_event_loader()
orbit_engine_root = Path.cwd().parent / 'orbit-engine'
stage6_dir = orbit_engine_root / 'data' / 'outputs' / 'rl_training' / 'stage6'

a4_events, d2_events = loader.load_latest_events(stage6_dir)
print(f"\n✅ Loaded {len(a4_events)} A4 events")

# 分析 A4 事件的特徵
neighbor_rsrp = []
serving_rsrp_from_event = []  # From event if available
trigger_margins = []
thresholds = []

for event in a4_events[:500]:  # Sample 500 events
    measurements = event.get('measurements', {})
    
    n_rsrp = measurements.get('neighbor_rsrp_dbm')
    threshold = measurements.get('threshold_dbm')
    margin = measurements.get('trigger_margin_db')
    
    # Try to get serving RSRP if available
    s_rsrp = measurements.get('serving_rsrp_dbm')
    
    if n_rsrp is not None:
        neighbor_rsrp.append(n_rsrp)
    if s_rsrp is not None:
        serving_rsrp_from_event.append(s_rsrp)
    if threshold is not None:
        thresholds.append(threshold)
    if margin is not None:
        trigger_margins.append(margin)

print(f"\n📊 A4 Event Analysis (N={len(a4_events)}):")
print(f"  Neighbor RSRP: min={np.min(neighbor_rsrp):.1f}, max={np.max(neighbor_rsrp):.1f}, mean={np.mean(neighbor_rsrp):.1f} dBm")
print(f"  Threshold: {np.unique(thresholds)} dBm")
print(f"  Trigger margin: min={np.min(trigger_margins):.1f}, max={np.max(trigger_margins):.1f}, mean={np.mean(trigger_margins):.1f} dB")
print(f"  Serving RSRP available in events: {len(serving_rsrp_from_event)}/{len(a4_events[:500])}")

if len(serving_rsrp_from_event) > 0:
    print(f"  Serving RSRP: min={np.min(serving_rsrp_from_event):.1f}, max={np.max(serving_rsrp_from_event):.1f}, mean={np.mean(serving_rsrp_from_event):.1f} dBm")
    rsrp_diffs = [n - s for n, s in zip(neighbor_rsrp[:len(serving_rsrp_from_event)], serving_rsrp_from_event)]
    print(f"  RSRP diff (N-S): min={np.min(rsrp_diffs):.1f}, max={np.max(rsrp_diffs):.1f}, mean={np.mean(rsrp_diffs):.1f} dB")

print("\n" + "=" * 80)
print("🚨 根本問題分析")
print("=" * 80)
print("""
觀察結果:
1. ✅ 新閾值已生效: threshold = -34.5 dBm (was -100.0)
2. ✅ Trigger margins 現在是真實的: 2-15 dB (was 55-80 dB)
3. ✅ 事件數量減少: 21,224 (was 48,002), 減少 55.8%

但是 100% 準確率依然存在，因為:

【數據洩漏的真正原因】
- A4 events 只包含 neighbor RSRP (沒有 serving RSRP)
- train_offline_bc_v3.py 生成 negative samples 的方式:
  * 隨機選擇 serving satellite 從 Stage 5
  * 隨機選擇 neighbor satellite 從 Stage 5
  * 計算 RSRP 差: neighbor_rsrp - serving_rsrp

問題:
- Handover events: neighbor RSRP >> threshold (-34.5 dBm)
                    所以 neighbor RSRP 都是高值 (-28 to -20 dBm)
- Maintain samples: 隨機選擇，包含所有 RSRP 範圍
                    如果 neighbor RSRP < threshold，margin 會是負的

但是！train_offline_bc_v3.py 使用的策略是:
  neighbor_rsrp = serving_rsrp - random(0, 3)
  
這讓 neighbor WORSE than serving，所以模型學到:
  if neighbor_rsrp > serving_rsrp: handover
  else: maintain
  
→ 100% accuracy

【正確的解決方案】
需要從 Stage 4 candidate pool 提取:
- Handover: neighbor satisfies threshold → 執行換手
- Maintain: neighbor is good but not good enough → 維持連線
  (例如: neighbor RSRP = -35.0 dBm, threshold = -34.5 dBm, margin = -0.5 dB)
""")
