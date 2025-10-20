# RL 算法替換指南

**當前實現**: DQN (Discrete Action Space)

**架構優勢**: Environment 完全符合 Gymnasium 標準 → 可替換各種算法

**最後更新**: 2025-10-20 (明确项目实施范围)

---

## ⚠️ IMPORTANT: PROJECT SCOPE CLARIFICATION

**本项目实施范围** (Phase 1-2, 2-3 weeks):
- ✅ **DQN** (已完成) - 唯一实施的 RL baseline
- ✅ **3 Rule-based baselines**: Strongest RSRP, A4-based Strategy, D2-based Strategy

**明确说明**:
- ❌ **Tier 2-3 算法（D3QN、PPO、A2C、Rainbow、SAC 等）不在当前项目实施范围内**
- ⭐ **未来工作**: 用户自己的算法 vs 以上 4 个 baselines (1 RL + 3 Rule-based)

**本文档用途**:
- ✅ Architecture reference (说明 Gymnasium 环境的可扩展性)
- ✅ Literature review context (了解领域研究现状)
- ❌ NOT a commitment to implement all listed algorithms

详见 [BASELINE_ALGORITHMS.md](BASELINE_ALGORITHMS.md) 完整说明

---

## 📚 2023-2025文獻調研更新

基於最新16篇論文的調研結果（**仅供了解领域研究现状，不代表本项目实施范围**）：

### Tier 1: 本项目实施
1. **DQN** ⭐⭐⭐⭐⭐ - 已完成 ✅
   - 使用頻率：62.5%的論文
   - 標準 RL baseline
   - **本项目唯一实施的 RL 算法**

### Tier 2: 文献调研参考（❌ 不在本项目实施范围）
2. **PPO** ⭐⭐⭐⭐⭐ - **Frontiers 2023明確推薦**
   - "PPO is the most stable algorithm, converging quickly"
   - 31.3%的論文使用
   - ❌ 不在本项目实施范围

3. **Double DQN** ⭐⭐⭐⭐
   - DQN的標準改進
   - ❌ 不在本项目实施范围

4. **Dueling DQN** ⭐⭐⭐⭐
   - ❌ 不在本项目实施范围

5. **D3QN** (Dueling + Double) ⭐⭐⭐⭐
   - ❌ 不在本项目实施范围

### Tier 3: 文献调研参考（❌ 不在本项目实施范围）
6. **A2C/A3C** ⭐⭐⭐ - ❌ 不在本项目实施范围
7. **SAC** ⭐⭐⭐ - ❌ 不在本项目实施范围

**项目实际 baselines**:
- 1 RL baseline: DQN
- 3 Rule-based baselines: Strongest RSRP, A4-based Strategy, D2-based Strategy

詳細文獻引用見 [BASELINE_ALGORITHMS.md](BASELINE_ALGORITHMS.md)

---

## 🎯 算法分類與難度

| 類別 | 算法 | 難度 | 時間 | Environment 修改 | Agent 修改 |
|------|------|------|------|-----------------|-----------|
| **Value-based** | Double DQN, Dueling DQN | ⭐ Easy | 1-2天 | ❌ 不需要 | ✅ 小改 |
| **Policy-based** | PPO, A2C | ⭐⭐ Medium | 3-5天 | ❌ 不需要 | ✅ 重寫 |
| **Continuous** | SAC, TD3 | ⭐⭐⭐ Medium-Hard | 5-7天 | ✅ 需要改 | ✅ 重寫 |
| **Model-based** | Dyna-Q, MBPO | ⭐⭐⭐⭐ Hard | 7-14天 | ✅ 需要改 | ✅ 重寫 |
| **Multi-agent** | MADDPG, QMIX | ⭐⭐⭐⭐⭐ Very Hard | 14-21天 | ✅ 大改 | ✅ 重寫 |

---

## ✅ 容易替換：Value-Based Algorithms

### 1. Double DQN

**修改內容**: 只改 Agent 的 train_step()

**原理**:
- DQN: 用同一個網絡選擇和評估動作 → 容易高估 Q 值
- Double DQN: 用 online network 選擇，target network 評估

**代碼修改** (`src/agents/double_dqn_agent.py`):

```python
# DQN 的更新
q_targets = rewards + gamma * target_q_values.max(1)[0]

# Double DQN 的更新
# 1. 用 online network 選擇最佳動作
best_actions = self.q_network(next_states).argmax(1)

# 2. 用 target network 評估該動作的 Q 值
q_targets = rewards + gamma * target_q_network(next_states).gather(1, best_actions)
```

**預期改進**: 減少 Q 值過估計，更穩定的訓練

**參考論文**: Deep Reinforcement Learning with Double Q-learning (AAAI 2016)

---

### 2. Dueling DQN

**修改內容**: 改 QNetwork 架構

**原理**: 分離 Value 和 Advantage
- V(s): 狀態價值（這個狀態本身有多好）
- A(s,a): 動作優勢（這個動作比其他動作好多少）
- Q(s,a) = V(s) + A(s,a)

**代碼修改** (`src/agents/dueling_dqn_network.py`):

```python
class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single value
        )

        # Advantage stream: A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # Per-action advantages
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values
```

**預期改進**: 更好地學習狀態價值，特別適合某些動作影響較小的場景

**參考論文**: Dueling Network Architectures for Deep RL (ICML 2016)

---

## ⚠️ 中等難度：Policy-Based Algorithms

### PPO (Proximal Policy Optimization)

**為什麼選擇 PPO？**
- ✅ 當前最流行的 policy gradient 方法
- ✅ 穩定、樣本效率高
- ✅ 適合衛星換手（連續決策）

**修改內容**:
1. 新增 Policy Network（輸出動作概率）
2. 新增 Value Network（估計狀態價值）
3. GAE (Generalized Advantage Estimation)
4. Clipped surrogate objective

**架構對比**:

```
DQN:
  Input → Q-Network → Q-values → argmax → Action

PPO:
  Input → Policy Network → Action Probabilities → Sample → Action
  Input → Value Network → V(s) → 用於計算 Advantage
```

**核心代碼** (`src/agents/ppo_agent.py`):

```python
class PPOAgent:
    def __init__(self, obs_space, action_space, config):
        self.policy = PolicyNetwork(...)  # 輸出動作概率
        self.value = ValueNetwork(...)    # 估計狀態價值

        self.clip_epsilon = 0.2  # PPO clipping parameter
        self.gae_lambda = 0.95   # GAE lambda

    def select_action(self, state):
        # 從 policy 輸出的概率分佈中採樣
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def train_step(self, trajectories):
        # 1. 計算 GAE advantages
        advantages = self.compute_gae(trajectories)

        # 2. PPO clipped objective
        ratio = new_probs / old_probs
        clipped_ratio = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)
        loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # 3. Update policy and value networks
        ...
```

**Training Loop 修改**:

```python
# DQN: 每步更新
for step in episode:
    action = agent.select_action(state)
    next_state, reward = env.step(action)
    agent.store_experience(...)
    agent.train_step()  # 每步都可以訓練

# PPO: 批次更新（需要完整軌跡）
trajectory = []
for step in episode:
    action, log_prob = agent.select_action(state)
    next_state, reward = env.step(action)
    trajectory.append((state, action, reward, log_prob))

# Episode 結束後才訓練
agent.train_step(trajectory)
```

**預估時間**: 3-5 天

**參考論文**: Proximal Policy Optimization Algorithms (arXiv 2017)

---

## ⚠️⚠️ 中等偏難：Continuous Action Space

### SAC (Soft Actor-Critic)

**為什麼難？**
- ❌ 衛星換手本質是離散動作（選擇哪顆衛星）
- ⚠️ 需要重新定義動作空間

**可能的連續動作定義**:

**選項 1**: 連續衛星 ID
```python
# 當前 (Discrete)
action_space = Discrete(11)  # 0-10: stay or switch to satellite 0-9

# 連續版本
action_space = Box(low=0, high=100, shape=(1,))  # 衛星 ID (連續)
# 問題：如何從連續值映射到實際衛星？
```

**選項 2**: 多維連續動作
```python
action_space = Box(low=-1, high=1, shape=(3,))
# [0]: 是否換手 (sigmoid → 0/1)
# [1-2]: 目標衛星特徵 (elevation, azimuth) → 找最接近的衛星
```

**問題**：
- 衛星換手天然是離散決策
- 強行改成連續可能降低性能
- 論文都用離散動作

**建議**: 除非有特殊需求，否則保持 Discrete action space

**如果真的要實現 SAC**:
- 修改 Environment: action_space 改為 Box
- 實現 action → satellite 映射
- SAC Agent: Actor-Critic + Entropy regularization
- 預估時間: 5-7 天

---

## 🎓 實用建議

### 推薦嘗試的算法（優先順序）

1. **Double DQN** (1天)
   - ✅ 簡單、有效
   - ✅ 減少 Q 值過估計
   - ✅ 與論文對比公平（都是 DQN 系列）

2. **Dueling DQN** (2天)
   - ✅ 架構改進
   - ✅ 可能提升性能
   - ✅ 適合換手問題（某些狀態下動作差異小）

3. **PPO** (3-5天)
   - ✅ 當前最流行
   - ✅ 穩定性好
   - ✅ 可以作為對比實驗（DQN vs PPO）

4. **Rainbow DQN** (2-3天)
   - ✅ 組合多個改進（Double, Dueling, Prioritized Replay, etc.）
   - ✅ SOTA performance
   - ⚠️ 複雜度高

### 不建議嘗試的算法

1. **Continuous algorithms (SAC, TD3)**
   - ❌ 換手本質是離散決策
   - ❌ 強行改成連續沒有明顯好處

2. **Model-based RL**
   - ❌ 需要學習軌道模型（我們已經有 SGP4）
   - ❌ 計算量大

3. **Multi-agent RL**
   - ❌ 問題定義不清楚（誰是 agent？）
   - ❌ 實現複雜度太高

---

## 🔧 快速實現步驟

### 實現 Double DQN (1天)

```bash
# 1. 複製當前 Agent
cp src/agents/dqn_agent_v2.py src/agents/double_dqn_agent.py

# 2. 修改 train_step() 方法
#    - 改用 online network 選擇動作
#    - 用 target network 評估 Q 值

# 3. 測試
python3 train_online_rl.py \
    --agent-type double_dqn \
    --num-episodes 100 \
    --output-dir output/double_dqn_test

# 4. 對比結果
#    - DQN vs Double DQN reward curves
#    - TensorBoard 對比
```

### 實現 PPO (3-5天)

```bash
# 1. 建立新 Agent
touch src/agents/ppo_agent.py

# 2. 實現核心組件
#    - PolicyNetwork (輸出動作概率)
#    - ValueNetwork (估計狀態價值)
#    - GAE 計算
#    - PPO loss

# 3. 修改 training loop
#    - 收集完整 trajectory
#    - Batch update

# 4. 測試
python3 train_online_rl.py \
    --agent-type ppo \
    --num-episodes 500 \
    --output-dir output/ppo_test
```

---

## 📚 參考資源

### 論文

1. **DQN**: Playing Atari with Deep Reinforcement Learning (DeepMind 2013)
2. **Double DQN**: Deep RL with Double Q-learning (AAAI 2016)
3. **Dueling DQN**: Dueling Network Architectures for Deep RL (ICML 2016)
4. **Rainbow**: Rainbow: Combining Improvements in Deep RL (AAAI 2018)
5. **PPO**: Proximal Policy Optimization Algorithms (arXiv 2017)

### 實現參考

1. **Stable-Baselines3**: https://github.com/DLR-RM/stable-baselines3
   - 高質量實現（DQN, PPO, SAC, TD3 等）
   - 可以參考接口設計

2. **CleanRL**: https://github.com/vwxyzjn/cleanrl
   - 簡潔、教學向的實現
   - 適合學習

3. **RLlib**: https://docs.ray.io/en/latest/rllib/
   - Ray 的 RL 庫
   - 支持分佈式訓練

---

## ✅ 總結

**當前架構的優勢**:
- ✅ Environment 完全符合 Gymnasium 標準
- ✅ 可以輕鬆替換 Value-based algorithms (1-2天) - 如果将来需要
- ✅ 可以實現 Policy-based algorithms (3-5天) - 如果将来需要
- ⚠️ Continuous algorithms 需要修改 Environment (5-7天) - 如果将来需要

**本项目实际实施** (Phase 1-2):
1. ✅ DQN baseline (已完成)
2. ✅ 3 Rule-based baselines: Strongest RSRP, A4-based Strategy, D2-based Strategy
3. ❌ 不实施其他 RL 算法（D3QN、PPO、A2C 等）

**实际论文实验**:
- RL baseline: DQN
- Rule-based baselines: Strongest RSRP, A4-based Strategy, D2-based Strategy
- 未来对比: 用户自己的算法 vs 以上 4 个 baselines

**本文档价值**: 说明 Gymnasium 架构的可扩展性，以及如何实现其他算法（如果将来需要）

---

**Date**: 2025-10-20
**Status**: ✅ 架構設計參考（實際項目只實施 DQN + 3 Rule-based baselines）
