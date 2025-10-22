# 診斷報告：100% 訓練準確率問題

## 執行日期
2025-10-21

## 問題描述
即使修正了 A3/A4/A5 閾值，BC 模型訓練仍然達到 100% 準確率，表明存在數據洩漏。

## 已完成的修復

### 1. ✅ 閾值配置Bug修復
**問題**: gpp_event_detector 收到嵌套配置但期望扁平結構
- Config: `gpp_events.a4.rsrp_threshold_dbm = -34.5`
- Detector 期望: `a4_threshold_dbm = -34.5`

**修復**: stage6_research_optimization_processor.py:169-228
- 新增 `_flatten_gpp_config()` 方法
- 轉換嵌套結構為扁平鍵值

**驗證**:
```
舊配置: A4 門檻 = -100.0 dBm
新配置: A4 門檻 = -34.5 dBm ✅
```

### 2. ✅ Trigger Margin 修正
**舊數據** (threshold = -100.0 dBm):
- Trigger margin: 55-80 dB (無變化)
- 所有事件 100% 觸發

**新數據** (threshold = -34.5 dBm):
- Trigger margin: 2.0 - 15.2 dB ✅
- Median: 4.5 dB
- 90th percentile: 9.0 dB
- 事件減少: 48,002 → 21,224 (55.8% 被過濾)

### 3. ✅ 數據驅動閾值
基於 48,002 真實換手事件分析:
- A3 offset: 2.5 dB (實測 RSRP 範圍優化)
- A4 threshold: -34.5 dBm (30th percentile)
- A5 threshold1: -36.0 dBm (10th percentile)
- A5 threshold2: -33.0 dBm (40th percentile)

## 🚨 根本問題：Negative Sampling 策略錯誤

### 當前策略 (train_offline_bc_v3.py)
```python
# Handover events (from Stage 6)
neighbor_rsrp = -28 to -20 dBm  # All high values (> threshold)

# Maintain samples (generated)
neighbor_rsrp = serving_rsrp - random(0, 3)  # Makes neighbor WORSE
```

### 結果
- **Handover**: neighbor RSRP > threshold → neighbor is GOOD
- **Maintain**: neighbor RSRP < serving RSRP → neighbor is BAD

模型學到簡單規則:
```
if neighbor_rsrp > serving_rsrp:
    return HANDOVER
else:
    return MAINTAIN
```
→ **100% accuracy**

## 💡 正確的解決方案

### 問題關鍵
真實的 maintain 場景應該是：
- Neighbor 信號**良好**，但**未達到換手閾值**
- 例如: neighbor RSRP = -35.0 dBm, threshold = -34.5 dBm
  → margin = -0.5 dB → **不觸發換手**

### 數據來源
需要從 **Stage 4 candidate pool** 提取:
- 包含所有可見的候選衛星
- 根據閾值條件標註:
  * `margin > 0` → handover (positive class)
  * `margin < 0` → maintain (negative class)

### 為什麼這樣才正確?
1. **現實場景**: 多顆衛星同時可見
2. **決策邊界**: 部分滿足閾值，部分不滿足
3. **學習任務**: 模型需學習閾值附近的細微差異

## 📊 數據統計對比

| 指標 | 舊閾值 | 新閾值 | 改善 |
|------|--------|--------|------|
| A4 threshold | -100.0 dBm | -34.5 dBm | ✅ |
| Trigger margin range | 55-80 dB | 2-15 dB | ✅ |
| A4 event count | 48,002 | 21,224 | 55.8% 減少 ✅ |
| Training accuracy | 100% | 100% | ❌ 未改善 |

## ⚠️ 為什麼閾值修正還不夠?

即使 trigger margins 現在是真實的 (2-15 dB)，但:
1. **Positive samples** (handover): 來自 Stage 6 events
   - neighbor RSRP 都是高值 (通過閾值)
2. **Negative samples** (maintain): 隨機生成
   - neighbor RSRP 隨機，與 threshold 無關
   - 使用 `serving - random(0,3)` 策略

這兩類樣本在特徵空間完全分離，無法創造學習難度。

## ✅ 下一步行動

### 短期 (修正負樣本生成)
1. 修改 train_offline_bc_v3.py
2. 從 Stage 4 candidate pool 提取
3. 使用 threshold 條件標註

### 中期 (數據 pipeline 重構)
1. Stage 6 輸出 candidate pool data
2. 包含所有可見衛星 + threshold calculations
3. 自動生成 handover/maintain pairs

### 長期 (強化學習)
1. 使用新數據訓練 BC policy
2. 部署到 RL environment
3. 通過 online learning 優化

## 📝 結論

**問題診斷**: ✅ 完成
- 根本原因: Negative sampling 策略使 neighbor 變差
- 數據洩漏: Handover (好鄰居) vs Maintain (壞鄰居) 完全分離

**閾值修正**: ✅ 成功
- Config bug 已修復
- Trigger margins 現在真實 (2-15 dB)
- 事件過濾正常 (55.8% 減少)

**訓練改善**: ❌ 尚未完成
- 需要重新設計 negative sampling
- 必須使用 Stage 4 candidate pool
- 預期準確率: 85-95% (修正後)

---
**生成時間**: 2025-10-21 01:32:00  
**數據來源**: Stage 6 RL training dataset (21,224 A4 events)  
**分析方法**: 特徵分布統計 + 訓練結果驗證  
