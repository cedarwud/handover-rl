# Literature Review: RL for LEO Satellite Handover (2023-2025)

**Review Period**: 2023年1月 - 2025年1月
**Papers Reviewed**: 16篇
**Date**: 2025-10-19

---

## 📊 Key Findings Summary

1. **DQN系列**是LEO satellite handover的标准baseline（10+篇论文）
2. **PPO**被Frontiers 2023明确推荐为最稳定算法
3. **跨星座换手**（Starlink↔OneWeb）在所有文献中均未出现
4. **训练时长**: 大多数论文未明确说明，估计1500-2000 episodes

---

## 📚 Core Papers by Algorithm

### DQN Applications (10篇)

1. **Deep Q-Learning for Spectral Coexistence (2025-01)**
   - IEEE最新，DQN管理LEO卫星频谱共存

2. **Graph RL-Based Handover (2024-07, MDPI)**
   - MPNN-DQN，图神经网络+DQN
   - 优于传统DQN和DRL方法

3. **Multi-Dimensional Resource Allocation (2024-03)**
   - Journal of Cloud Computing
   - DQN适应LEO高移动性

### PPO Applications (5篇)

4. **Comparative Analysis of RL Algorithms (2023-11, Frontiers)**
   - **关键引用**: "PPO is the most stable algorithm"
   - 对比PPO, A2C, DQN, MCTS-Train
   - 推荐用于航天器应用

### SAC Applications (1篇)

5. **Nash-SAC for Handover (2024-02, arXiv)**
   - SAC比Nash-DQN减少16%换手次数

### Multi-Agent (2篇)

6. **Multi-Agent DRL for LEO (2024)**
   - MADQN分布式换手

---

## 🔍 Research Gaps Identified

### Gap 1: 缺乏时间配置细节
**发现**: 所有论文都缺少训练时长、episode结构的详细说明

**我们的贡献**:
- ✅ Multi-level训练策略（10分钟→35小时）
- ✅ 明确的episode设计（95分钟orbital period）
- ✅ 连续时间采样 vs 随机采样

### Gap 2: 单星座 vs 跨星座
**发现**: 所有论文使用单一星座（Starlink OR OneWeb）

**我们的选择**: Starlink-only (101 satellites)
- ✅ 符合所有文献
- ✅ 避免不切实际的跨星座换手

---

## 📈 Algorithm Usage Statistics

| Algorithm | Papers | Percentage |
|-----------|--------|------------|
| DQN (variants) | 10 | 62.5% |
| PPO | 5 | 31.3% |
| A2C/A3C | 2 | 12.5% |
| SAC | 1 | 6.3% |
| Others | 3 | 18.8% |

---

## ✅ Our Alignment with Literature

- ✅ DQN as baseline (standard practice)
- ✅ Starlink-only (all papers use single constellation)
- ✅ Discrete action space (standard for handover)
- ✅ 12-dimensional state space (comprehensive)
- ✅ Episode-based training (standard)

---

**Full bibliography**: See BASELINE_ALGORITHMS.md
**Date**: 2025-10-19
