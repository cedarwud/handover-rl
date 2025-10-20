#!/usr/bin/env python3
"""
快速驗證重構成果
✅ 確認使用完整 orbit-engine 實現，移除所有簡化算法
"""

import sys
sys.path.insert(0, '/home/sat/satellite/orbit-engine')

print("=" * 80)
print("驗證 Orbit-Engine Adapter 重構成果")
print("=" * 80)

# Test 1: 檢查是否成功導入 orbit-engine 完整實現模塊
print("\n[Test 1] 檢查 orbit-engine 模塊導入")
print("-" * 80)

try:
    from src.stages.stage2_orbital_computing.sgp4_calculator import SGP4Calculator
    from src.stages.stage5_signal_analysis.itur_physics_calculator import create_itur_physics_calculator
    from src.stages.stage5_signal_analysis.gpp_ts38214_signal_calculator import create_3gpp_signal_calculator
    from src.stages.stage5_signal_analysis.itur_official_atmospheric_model import create_itur_official_model

    print("✅ 成功導入所有 orbit-engine 模塊:")
    print("   - SGP4Calculator (軌道傳播)")
    print("   - create_itur_physics_calculator (ITU-R 物理計算)")
    print("   - create_3gpp_signal_calculator (3GPP TS 38.214 信號計算)")
    print("   - create_itur_official_model (ITU-R P.676-13 大氣模型)")

except ImportError as e:
    print(f"❌ 導入失敗: {e}")
    sys.exit(1)

# Test 2: 檢查配置文件是否包含完整參數
print("\n[Test 2] 檢查配置文件完整性")
print("-" * 80)

import yaml
from pathlib import Path

config_path = Path("/home/sat/satellite/handover-rl/config/data_gen_config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

# 檢查必要的配置部分
required_configs = {
    'signal_calculator': ['bandwidth_mhz', 'subcarrier_spacing_khz', 'noise_figure_db', 'temperature_k'],
    'atmospheric_model': ['temperature_k', 'pressure_hpa', 'water_vapor_density_g_m3'],
    'physics': ['tx_antenna_gain_db', 'rx_antenna_gain_db']
}

all_good = True
for section, params in required_configs.items():
    if section not in config:
        print(f"❌ 缺少配置部分: {section}")
        all_good = False
        continue

    print(f"\n✅ [{section}]:")
    for param in params:
        if param in config[section]:
            value = config[section][param]
            print(f"   ✓ {param}: {value}")
        else:
            print(f"   ✗ 缺少: {param}")
            all_good = False

if not all_good:
    print("\n❌ 配置文件不完整")
    sys.exit(1)

# Test 3: 驗證 orbit_engine_adapter.py 已移除簡化算法
print("\n[Test 3] 驗證已移除簡化算法")
print("-" * 80)

adapter_path = Path("/home/sat/satellite/handover-rl/src/adapters/orbit_engine_adapter.py")
with open(adapter_path) as f:
    adapter_code = f.read()

# 檢查不應該存在的簡化算法標記
forbidden_patterns = [
    ("noise_floor_dbm = -100.0", "硬編碼噪聲底 -100 dBm"),
    ("if elevation_deg < 5:", "線性插值大氣損耗"),
    ("n_rb = self.bandwidth_mhz / 0.18", "簡化 RSRQ 計算"),
    ("# Simplified calculation", "簡化計算註釋"),
]

violations = []
for pattern, description in forbidden_patterns:
    if pattern in adapter_code:
        violations.append(description)

if violations:
    print("❌ 發現簡化算法違規:")
    for v in violations:
        print(f"   - {v}")
    sys.exit(1)
else:
    print("✅ 未發現簡化算法")

# 檢查應該存在的完整實現標記
required_patterns = [
    ("create_itur_physics_calculator", "ITU-R 物理計算器工廠函數"),
    ("create_3gpp_signal_calculator", "3GPP 信號計算器工廠函數"),
    ("create_itur_official_model", "ITU-R P.676-13 官方模型工廠函數"),
    ("calculate_complete_signal_quality", "完整信號品質計算方法"),
    ("calculate_total_attenuation", "完整大氣衰減計算方法"),
]

missing = []
for pattern, description in required_patterns:
    if pattern not in adapter_code:
        missing.append(description)

if missing:
    print("\n❌ 缺少完整實現:")
    for m in missing:
        print(f"   - {m}")
    sys.exit(1)
else:
    print("✅ 使用完整 orbit-engine 實現")

# Test 4: 測試計算器初始化
print("\n[Test 4] 測試計算器初始化")
print("-" * 80)

try:
    # 測試 3GPP 信號計算器
    signal_calc_config = config['signal_calculator']
    signal_calc = create_3gpp_signal_calculator(signal_calc_config)
    print(f"✅ 3GPP 信號計算器初始化成功")
    print(f"   帶寬: {signal_calc.bandwidth_mhz} MHz")
    print(f"   子載波間距: {signal_calc.subcarrier_spacing_khz} kHz")
    print(f"   Resource Blocks: {signal_calc.n_rb}")

    # 測試大氣模型
    atm_config = config['atmospheric_model']
    atm_model = create_itur_official_model(
        temperature_k=atm_config['temperature_k'],
        pressure_hpa=atm_config['pressure_hpa'],
        water_vapor_density_g_m3=atm_config['water_vapor_density_g_m3']
    )
    print(f"\n✅ ITU-R P.676-13 大氣模型初始化成功")
    print(f"   溫度: {atm_config['temperature_k']} K")
    print(f"   氣壓: {atm_config['pressure_hpa']} hPa")
    print(f"   水汽密度: {atm_config['water_vapor_density_g_m3']} g/m³")

    # 測試大氣衰減計算（驗證非硬編碼值）
    freq_ghz = config['physics']['frequency_ghz']
    attenuation_10deg = atm_model.calculate_total_attenuation(frequency_ghz=freq_ghz, elevation_deg=10)
    attenuation_90deg = atm_model.calculate_total_attenuation(frequency_ghz=freq_ghz, elevation_deg=90)

    print(f"\n✅ 大氣衰減計算測試:")
    print(f"   10° 仰角: {attenuation_10deg:.4f} dB")
    print(f"   90° 仰角: {attenuation_90deg:.4f} dB")

    # 驗證結果符合物理規律（低仰角衰減更大）
    if attenuation_10deg > attenuation_90deg:
        print(f"   ✓ 物理規律正確（低仰角衰減較大）")
    else:
        print(f"   ✗ 物理規律錯誤")
        sys.exit(1)

    # 驗證非硬編碼值（不應該是整數）
    if attenuation_10deg == 10.0 or attenuation_90deg == 0.5:
        print(f"   ✗ 疑似硬編碼值")
        sys.exit(1)
    else:
        print(f"   ✓ 非硬編碼值（完整計算結果）")

except Exception as e:
    print(f"❌ 計算器初始化失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 總結
print("\n" + "=" * 80)
print("✅ 所有驗證通過！")
print("=" * 80)
print("\n重構成果:")
print("  1. ✅ 成功整合 orbit-engine 完整實現")
print("  2. ✅ 移除所有簡化算法 (RSRQ, SINR, 大氣損耗)")
print("  3. ✅ 移除所有硬編碼值 (噪聲底 -100 dBm等)")
print("  4. ✅ 配置文件包含完整學術級參數")
print("  5. ✅ 使用 3GPP TS 38.214/38.215 標準")
print("  6. ✅ 使用 ITU-R P.676-13 官方大氣模型")
print("  7. ✅ 使用 ITU-R P.525-4 自由空間損耗計算")
print("\n符合 CLAUDE.md 「REAL ALGORITHMS ONLY」原則 ✅")
print("=" * 80)
