# 3GPP 換手事件閾值配置建議

## 📚 資料來源總結

### 1. 3GPP 標準規範
- **3GPP TS 38.331 v18.5.1**: 定義事件公式，但不指定具體參數值
- **3GPP TR 38.821 v16.0.0**: NTN 技術報告，建議基於場景調整參數

### 2. 學術文獻發現

#### 地面網絡典型值 (LTE/5G NR)
根據搜尋結果和學術文獻：

**A3 事件**:
- A3 Offset: **2-4 dB** (典型值 3 dB)
- Hysteresis: **1-3 dB** (典型值 2 dB)
- Time-to-Trigger: 40-480 ms
- 規則: `a3_offset > hysteresis` 以避免 ping-pong

**A4 事件**:
- RSRP Threshold: **-110 到 -95 dBm** (地面網絡)
- 典型值: -100 到 -105 dBm
- Hysteresis: 1-3 dB

**A5 事件**:
- Threshold1 (Serving): -110 到 -100 dBm
- Threshold2 (Neighbor): -95 到 -85 dBm

### 3. LEO NTN 特殊考量

**關鍵差異**:
1. **RSRP 範圍更窄**: LEO 衛星 RSRP 變化範圍約 15-20 dB (vs 地面網絡 60 dB)
2. **信號更強**: LEO 典型 RSRP 在 -45 到 -20 dBm (vs 地面 -110 到 -50 dBm)
3. **快速移動**: 衛星移動速度快，需要更靈敏的換手

**調整原則**:
- Offset/Hysteresis 需要**等比例縮小**: `地面值 × (LEO_RSRP_range / 地面_RSRP_range)`
- Threshold 需要**向上調整**至實際 RSRP 範圍

---

## 🎯 基於實測數據的建議配置

### 當前數據統計 (orbit-engine Stage 5/6)
```
RSRP 範圍: -44.84 到 -19.30 dBm (範圍 25.5 dB)
RSRP 平均: -32.77 dBm
RSRP 中位數: -33.02 dBm
標準差: ~5.89 dB (根據 Stage 6 config 註釋)
```

---

## 📋 推薦配置

### Option 1: 保守配置 (較少換手，穩定優先)

```yaml
gpp_events:
  a3:
    offset_db: 3.0           # 略高於當前 2.0
    hysteresis_db: 2.0       # 增加穩定性
    time_to_trigger_ms: 160  # 延長觀察時間

  a4:
    rsrp_threshold_dbm: -36.0  # 10th percentile
    hysteresis_db: 2.0
    time_to_trigger_ms: 160

  a5:
    rsrp_threshold1_dbm: -42.0  # Serving 劣化門檻 (5th percentile)
    rsrp_threshold2_dbm: -32.0  # Neighbor 良好門檻 (60th percentile)
    hysteresis_db: 2.0
    time_to_trigger_ms: 160

  d2:
    starlink:
      d2_threshold1_km: 1400.0  # 接近 serving 平均 (1494 km)
      d2_threshold2_km: 700.0   # 接近 neighbor 中位數 (695 km)
      hysteresis_km: 100.0
```

**預期結果**:
- A3: 約 70-80% 事件觸發 (RSRP diff > 5.0 dB)
- A4: 約 85-90% 觸發 (鄰居 RSRP > -36 dBm)
- 較少 ping-pong，換手穩定

---

### Option 2: 平衡配置 (推薦)

```yaml
gpp_events:
  a3:
    offset_db: 2.5           # 介於 2.0-3.0
    hysteresis_db: 1.5       # 保持當前值
    time_to_trigger_ms: 100  # 保持當前值

  a4:
    rsrp_threshold_dbm: -34.5  # 30th percentile (如分析建議)
    hysteresis_db: 2.0
    time_to_trigger_ms: 100

  a5:
    rsrp_threshold1_dbm: -40.0  # 略低於當前 -41.0
    rsrp_threshold2_dbm: -33.0  # 略高於當前 -34.0
    hysteresis_db: 2.0
    time_to_trigger_ms: 100

  d2:
    starlink:
      d2_threshold1_km: 1200.0  # 允許更多 D2 事件
      d2_threshold2_km: 900.0   # 鄰居需明顯較近
      hysteresis_km: 80.0
```

**預期結果**:
- A3: 約 50-60% 事件觸發
- A4: 約 65-75% 觸發
- 平衡換手頻率和穩定性
- **更適合 ML 訓練** (有足夠的正負樣本)

---

### Option 3: 激進配置 (快速換手，性能優先)

```yaml
gpp_events:
  a3:
    offset_db: 2.0           # 保持當前值
    hysteresis_db: 1.0       # 降低門檻
    time_to_trigger_ms: 64   # 最快響應

  a4:
    rsrp_threshold_dbm: -33.0  # Median
    hysteresis_db: 1.5
    time_to_trigger_ms: 64

  a5:
    rsrp_threshold1_dbm: -38.0  # 更早觸發換手
    rsrp_threshold2_dbm: -31.0  # 要求鄰居更好
    hysteresis_db: 1.5
    time_to_trigger_ms: 64

  d2:
    starlink:
      d2_threshold1_km: 1000.0
      d2_threshold2_km: 1100.0
      hysteresis_km: 50.0
```

**預期結果**:
- A3: 約 30-40% 事件觸發
- A4: 約 40-50% 觸發
- 更快換手，可能增加 ping-pong 風險

---

## 🔬 驗證方法

### 1. 檢查觸發率分佈

修改配置後，運行：
```bash
python verify_all_thresholds.py
```

**理想分佈**:
- 20-80% 的樣本應該**不觸發**換手（maintain 決策）
- 這樣 ML 模型才需要真正學習決策邊界

### 2. 檢查數據洩漏

```python
# A4 事件：Trigger margin 應有變化
trigger_margins = [event['measurements']['trigger_margin_db'] for event in a4_events]
print(f"Trigger margin range: {min(trigger_margins)} to {max(trigger_margins)} dB")
# ✅ 應該看到負值或小正值，不應全是 55-80 dB

# A3 事件：RSRP diff 應有 below threshold 的樣本
rsrp_diffs = [neighbor_rsrp - serving_rsrp for event in a3_events]
below_threshold = sum(1 for d in rsrp_diffs if d < (offset + hysteresis))
print(f"Below threshold: {below_threshold}/{len(rsrp_diffs)} ({below_threshold/len(rsrp_diffs)*100:.1f}%)")
# ✅ 應該有 10-30% 樣本低於閾值
```

### 3. 重新訓練模型

期望結果：
- **Training accuracy**: 85-95% (not 100%)
- **Test accuracy**: 80-92% (略低於訓練集)
- **Loss**: 應持續下降，不應立即到 0.0000

---

## 📚 參考資料

1. **3GPP TS 38.331 v18.5.1 Section 5.5.4**: A3/A4/A5 事件定義
2. **3GPP TR 38.821 v16.0.0 Section 6.4.3**: NTN 場景換手建議
3. **地面網絡經驗**:
   - A3 offset: 2-4 dB (典型 3 dB)
   - Hysteresis: 1-3 dB (典型 2 dB)
   - Rule: offset > hysteresis 避免 ping-pong
4. **LEO NTN 調整係數**:
   - RSRP 範圍比例: 25.5/60 ≈ 0.42
   - 建議 offset/hysteresis 縮小 50-60%

---

## 🎯 最終建議

**對於 ML 訓練任務，強烈推薦 Option 2 (平衡配置)**：

### 理由：
1. **避免數據洩漏**: A4 threshold 調高到 -34.5 dBm，trigger margin 會有正負值變化
2. **平衡樣本**: 約 50% 觸發率，positive/negative samples 更均衡
3. **學習難度適中**: 模型需要學習真實決策邊界，不是簡單閾值
4. **符合 3GPP 精神**: offset > hysteresis，避免 ping-pong

### 下一步：
1. 修改 `orbit-engine/config/stage6_research_optimization_config.yaml`
2. 重新生成 Stage 6 數據: `cd orbit-engine && ./run.sh --stage 6`
3. 重新訓練 BC 模型: `python train_offline_bc_v2.py`
4. 驗證準確率降到合理範圍 (85-95%)
