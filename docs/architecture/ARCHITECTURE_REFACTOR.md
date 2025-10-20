# 模块化RL架构重构设计

**目标**: 从 DQN-only 架构 → 支持多种RL算法的通用框架

**创建时间**: 2025-10-19
**状态**: 🚧 设计阶段

---

## ⚠️ IMPORTANT: PROJECT SCOPE CLARIFICATION

**本项目实施范围** (Phase 1-2 only, 2-3 weeks):
- ✅ **Phase 1**: DQN refactoring to modular architecture
- ✅ **Phase 2**: Verify architecture supports multiple algorithms (proof of concept)
- ✅ **Baseline Framework**: 1 RL baseline (DQN) + 3 Rule-based baselines

**明确说明**:
- ❌ **Phase 3-4 (D3QN, PPO, A2C) 不在当前项目实施范围内**
- ❌ **不需要实现其他 RL 算法**（D3QN、PPO、A2C、Rainbow、SAC 等）
- ⭐ **未来工作**: 用户自己的算法 vs 以上 4 个 baselines

**本文档用途**:
- ✅ Architecture design reference (如何设计可扩展架构)
- ✅ Phase 1-2 implementation guide (DQN 重构 + 架构验证)
- ❌ NOT a commitment to implement all listed algorithms

详见 [BASELINE_ALGORITHMS.md](../algorithms/BASELINE_ALGORITHMS.md) 完整说明

---

## ⚠️ 核心功能保留清单（P0 - CRITICAL）

**重要**: 以下功能是系统的核心创新点（README.md "Novel Aspects"），重构时**必须完整保留**。

### 1. Multi-Level Training Strategy ⭐⭐⭐⭐⭐

**重要性**: README.md 列为 "Novel Aspects #1" - 研究贡献的第一个创新点

**现有功能**:
- 6 个训练级别（Level 0-5）
- 渐进式验证策略：10分钟 → 35小时
- 快速迭代开发（Level 1: 2小时 vs Level 5: 35小时）

**必须保留**:
```bash
# 重构后必须支持
train.py --algorithm dqn --level 0  # 10分钟 - Smoke test
train.py --algorithm dqn --level 1  # 2小时 - Quick validation
train.py --algorithm dqn --level 3  # 10小时 - Validation (推荐)
train.py --algorithm dqn --level 5  # 35小时 - Full training
```

**实现要求**:
- [ ] 创建 `src/configs/training_levels.py` 或 `config/levels/*.yaml`
- [ ] train.py 必须支持 `--level {0,1,2,3,4,5}` 参数
- [ ] 所有 6 个 level 的配置必须准确（卫星数、episodes、时间）
- [ ] 所有算法（DQN, PPO, A2C）都必须支持所有 level

**学术价值**:
- 这是论文的一个独立贡献点
- 展示如何高效进行 RL 实验
- 其他研究者可以复制此方法

**验证标准**:
- ✅ Level 1 训练时间 ≈ 2小时
- ✅ Level 3 训练时间 ≈ 10小时
- ✅ Level 5 训练时间 ≈ 35小时
- ✅ 所有算法在所有 level 都可运行

---

### 2. Gymnasium Environment ✅

**现有功能**: SatelliteHandoverEnv (已算法无关)

**保留状态**: ✅ 已完成，无需修改

---

### 3. Real TLE Data + Complete Physics ✅

**现有功能**:
- Space-Track.org TLE 数据
- ITU-R + 3GPP 完整物理模型
- 无简化算法、无 mock 数据

**保留状态**: ✅ 已完成，重构不影响

---

## 🎯 重构目标

### 从单一算法到多算法框架

**当前架构** (DQN-only):
```python
train_online_rl.py
├── 硬编码 DQNAgentV2
├── 硬编码 per-step update (off-policy)
├── 硬编码 epsilon-greedy exploration
└── 硬编码 experience replay buffer
```

**目标架构** (Algorithm-Agnostic):
```python
train.py --algorithm {dqn|ppo|a2c|sac}
├── Trainer Layer (Off-policy / On-policy)
├── Agent Layer (DQN / PPO / A2C / SAC)
└── Environment Layer (Gymnasium ✅ Already done)
```

---

## 🏗️ 架构设计

### 三层解耦架构

```
┌─────────────────────────────────────────┐
│  Layer 1: Training Entry Point          │
│  train.py (统一入口)                     │
│  ├── --algorithm {dqn, ppo, a2c}        │
│  ├── --level {0,1,2,3,4,5}              │
│  └── --config algorithms/xxx_config.yaml│
└──────────────────┬──────────────────────┘
                   ↓
┌──────────────────┴──────────────────────┐
│  Layer 2: Trainer (训练逻辑)            │
│  ├── OffPolicyTrainer (DQN, SAC, ...)  │
│  │   - Experience replay buffer         │
│  │   - Per-step update                  │
│  │   - Target network sync              │
│  └── OnPolicyTrainer (PPO, A2C, ...)   │
│      - Trajectory collection            │
│      - Batch update after episode       │
│      - GAE advantage calculation        │
└──────────────────┬──────────────────────┘
                   ↓
┌──────────────────┴──────────────────────┐
│  Layer 3: Agent (算法实现)              │
│  ├── BaseAgent (统一接口)               │
│  ├── DQNAgent                            │
│  ├── DoubleDQNAgent                      │
│  ├── DuelingDQNAgent                     │
│  ├── D3QNAgent                           │
│  ├── PPOAgent                            │
│  ├── A2CAgent                            │
│  └── SACAgent (future)                  │
└──────────────────┬──────────────────────┘
                   ↓
┌──────────────────┴──────────────────────┐
│  Layer 4: Environment                   │
│  SatelliteHandoverEnv (Gymnasium)       │
│  ✅ Already algorithm-agnostic          │
└─────────────────────────────────────────┘
```

---

## 📐 核心组件设计

### 1. BaseAgent 接口 (src/agents/base_agent.py)

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np

class BaseAgent(ABC):
    """
    所有RL Agent的统一接口

    设计原则:
    - 统一方法名 (select_action, update, save, load)
    - 灵活参数签名 (不同算法的update可以有不同参数)
    - 标准返回格式
    """

    @abstractmethod
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        选择动作

        Args:
            state: 观测状态 (shape: [12] for satellite handover)
            deterministic: 是否使用确定性策略
                - True: 评估模式，选择最优动作
                - False: 训练模式，探索式选择

        Returns:
            action: 动作索引 (0 or 1 for binary handover)

        NOTE: 不同算法的实现：
        - DQN: epsilon-greedy exploration
        - PPO: sample from policy distribution
        - A2C: sample from policy distribution
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> float:
        """
        更新Agent

        NOTE: 不同算法的update签名不同，这是合理的：
        - DQN: update(state, action, reward, next_state, done)
              或 update(batch) for batch update
        - PPO: update(trajectory: List[Dict])
        - A2C: update(trajectory: List[Dict])

        Returns:
            loss: 训练损失
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """保存模型到文件"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """从文件加载模型"""
        pass

    # 可选的回调方法（子类可以覆盖）
    def on_episode_start(self) -> None:
        """Episode开始时的回调"""
        pass

    def on_episode_end(self, episode_reward: float) -> None:
        """Episode结束时的回调"""
        pass

    def get_config(self) -> Dict[str, Any]:
        """返回Agent配置（用于保存/记录）"""
        return {}
```

### 2. OffPolicyTrainer (src/trainers/off_policy_trainer.py)

```python
class OffPolicyTrainer:
    """
    Off-policy算法的训练器 (DQN, SAC, etc.)

    特点:
    - 使用 experience replay buffer
    - Per-step update (每步都可以更新)
    - 可以从旧经验学习
    """

    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.config = config
        self.replay_buffer = ReplayBuffer(config['buffer_size'])

    def train_episode(self, episode_idx: int) -> Dict[str, float]:
        """
        训练一个episode

        Returns:
            metrics: {'reward': float, 'loss': float, 'handovers': int, ...}
        """
        obs, info = self.env.reset()
        episode_reward = 0
        episode_loss = 0
        num_updates = 0

        while True:
            # 1. Agent选择动作
            action = self.agent.select_action(obs, deterministic=False)

            # 2. 环境交互
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # 3. 存储到replay buffer
            self.replay_buffer.add(obs, action, reward, next_obs, done)

            # 4. Per-step update (off-policy特性)
            if len(self.replay_buffer) >= self.config['min_buffer_size']:
                batch = self.replay_buffer.sample(self.config['batch_size'])
                loss = self.agent.update(batch)
                episode_loss += loss
                num_updates += 1

            episode_reward += reward
            obs = next_obs

            if done:
                break

        avg_loss = episode_loss / num_updates if num_updates > 0 else 0

        return {
            'reward': episode_reward,
            'loss': avg_loss,
            'handovers': info.get('num_handovers', 0),
            'avg_rsrp': info.get('avg_rsrp', 0)
        }
```

### 3. OnPolicyTrainer (src/trainers/on_policy_trainer.py)

```python
class OnPolicyTrainer:
    """
    On-policy算法的训练器 (PPO, A2C, etc.)

    特点:
    - 收集完整trajectory
    - Episode结束后才更新
    - 只从当前策略收集的经验学习
    """

    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.config = config

    def train_episode(self, episode_idx: int) -> Dict[str, float]:
        """
        训练一个episode

        Returns:
            metrics: {'reward': float, 'loss': float, ...}
        """
        obs, info = self.env.reset()
        trajectory = []
        episode_reward = 0

        while True:
            # 1. Agent选择动作（同时返回log_prob用于PPO）
            action, log_prob = self.agent.select_action(obs, deterministic=False)

            # 2. 环境交互
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # 3. 存储到trajectory
            trajectory.append({
                'obs': obs,
                'action': action,
                'reward': reward,
                'log_prob': log_prob,  # PPO需要
                'done': done
            })

            episode_reward += reward
            obs = next_obs

            if done:
                break

        # 4. Episode结束后才更新 (on-policy特性)
        loss = self.agent.update(trajectory)

        return {
            'reward': episode_reward,
            'loss': loss,
            'handovers': info.get('num_handovers', 0),
            'avg_rsrp': info.get('avg_rsrp', 0)
        }
```

### 4. 统一训练入口 (train.py)

```python
import argparse
from pathlib import Path
import yaml

from src.trainers import OffPolicyTrainer, OnPolicyTrainer
from src.agents import DQNAgent, DoubleDQNAgent, PPOAgent, A2CAgent
from src.environments import SatelliteHandoverEnv
from src.utils import load_stage4_satellites, setup_logging

# 算法注册表
ALGORITHM_REGISTRY = {
    'dqn': {
        'agent_class': DQNAgent,
        'trainer_class': OffPolicyTrainer,
        'config_file': 'config/algorithms/dqn_config.yaml'
    },
    'double_dqn': {
        'agent_class': DoubleDQNAgent,
        'trainer_class': OffPolicyTrainer,
        'config_file': 'config/algorithms/double_dqn_config.yaml'
    },
    'ppo': {
        'agent_class': PPOAgent,
        'trainer_class': OnPolicyTrainer,
        'config_file': 'config/algorithms/ppo_config.yaml'
    },
    'a2c': {
        'agent_class': A2CAgent,
        'trainer_class': OnPolicyTrainer,
        'config_file': 'config/algorithms/a2c_config.yaml'
    },
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str,
                       choices=['dqn', 'double_dqn', 'ppo', 'a2c'],
                       required=True,
                       help='RL algorithm to use')
    parser.add_argument('--level', type=int, choices=[0,1,2,3,4,5],
                       default=1,
                       help='Training level (0=smoke test, 5=full training)')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # 1. 加载配置
    algo_info = ALGORITHM_REGISTRY[args.algorithm]
    with open(algo_info['config_file']) as f:
        algo_config = yaml.safe_load(f)

    with open('config/training_levels.yaml') as f:
        level_config = yaml.safe_load(f)[f'level_{args.level}']

    # 2. 创建Environment (算法无关)
    satellite_ids = load_stage4_satellites(
        constellation_filter='starlink',
        num_satellites=level_config['num_satellites']
    )

    env = SatelliteHandoverEnv(
        satellite_ids=satellite_ids,
        overlap=level_config['overlap']
    )

    # 3. 创建Agent (算法特定)
    AgentClass = algo_info['agent_class']
    agent = AgentClass(
        obs_space=env.observation_space,
        action_space=env.action_space,
        config=algo_config
    )

    # 4. 创建Trainer (根据算法类型)
    TrainerClass = algo_info['trainer_class']
    trainer = TrainerClass(env, agent, algo_config)

    # 5. 训练循环 (统一)
    logger = setup_logging(args.output_dir)

    for episode in range(level_config['num_episodes']):
        metrics = trainer.train_episode(episode)

        if (episode + 1) % level_config['log_interval'] == 0:
            logger.info(f"Episode {episode+1}: reward={metrics['reward']:.2f}, "
                       f"loss={metrics['loss']:.4f}")

        # Checkpoint saving
        if (episode + 1) % level_config['checkpoint_interval'] == 0:
            agent.save(f"{args.output_dir}/checkpoints/ep{episode+1}.pth")

    # 保存最终模型
    agent.save(f"{args.output_dir}/final_model.pth")
    print(f"✅ Training completed: {level_config['num_episodes']} episodes")

if __name__ == '__main__':
    main()
```

---

## 📁 目录结构重构

### 新的代码组织

```
src/
├── agents/                       # RL算法实现
│   ├── base_agent.py            # ✅ BaseAgent接口
│   ├── dqn/                     # DQN系列
│   │   ├── __init__.py
│   │   ├── dqn_agent.py         # ✅ 标准DQN
│   │   ├── double_dqn.py        # Double DQN
│   │   ├── dueling_dqn.py       # Dueling DQN
│   │   └── d3qn.py              # Dueling Double DQN
│   ├── ppo/                     # PPO系列
│   │   ├── __init__.py
│   │   ├── ppo_agent.py         # PPO
│   │   └── networks.py          # Policy/Value networks
│   ├── a2c/                     # A2C系列
│   │   ├── __init__.py
│   │   └── a2c_agent.py         # A2C
│   └── sac/                     # SAC (future)
│       └── sac_agent.py
├── trainers/                    # 训练逻辑 (新建)
│   ├── __init__.py
│   ├── off_policy_trainer.py   # Off-policy训练器
│   └── on_policy_trainer.py    # On-policy训练器
├── environments/                # 环境 (已完成)
│   └── satellite_handover_env.py
└── utils/                       # 工具函数
    ├── satellite_utils.py
    └── logging_utils.py

config/
├── training_levels.yaml         # 训练层级配置 (算法无关)
└── algorithms/                  # 算法配置 (新建)
    ├── dqn_config.yaml
    ├── double_dqn_config.yaml
    ├── ppo_config.yaml
    └── a2c_config.yaml

train.py                         # 统一训练入口 (新建)
train_online_rl.py              # 旧的DQN-only入口 (保留用于兼容)
```

---

## 🔄 重构策略

### Phase 1: DQN重构 (不破坏现有功能)

**目标**: 将现有DQN迁移到新架构，验证功能一致性

**步骤**:
1. 创建 `BaseAgent` 接口
2. 创建 `OffPolicyTrainer`
3. 将现有 `DQNAgentV2` 重构为 `DQNAgent` (继承BaseAgent)
4. 创建新的 `train.py --algorithm dqn`
5. 验证训练结果与 `train_online_rl.py` 一致

**验证标准**:
- ✅ 相同seed下reward曲线一致
- ✅ 相同超参数下loss下降趋势一致
- ✅ 可以加载旧模型继续训练

**时间**: 1-2天

### Phase 2: PPO实现 (验证架构扩展性)

**目标**: 实现PPO，验证新架构真正支持多算法

**步骤**:
1. 创建 `OnPolicyTrainer`
2. 实现 `PPOAgent`
   - PolicyNetwork (输出动作概率)
   - ValueNetwork (估计状态价值)
   - GAE计算
   - PPO loss (clipped surrogate objective)
3. 创建 `config/algorithms/ppo_config.yaml`
4. Level 1对比实验 (DQN vs PPO, 100 episodes)

**验证标准**:
- ✅ PPO训练收敛（loss下降）
- ✅ PPO reward有提升趋势
- ✅ 可以用相同的 `train.py` 入口

**时间**: 3-5天

### ~~Phase 3: DQN系列扩展~~ ❌ NOT IN CURRENT PROJECT SCOPE

**⚠️ 重要说明**: Phase 3-4 不在当前项目实施范围内

**项目范围**: 1 RL baseline (DQN) + 3 Rule-based baselines only
- ❌ 不实现 Double DQN, Dueling DQN, D3QN
- ❌ 不实现 PPO, A2C, SAC
- ⭐ 未来工作：用户自己的算法 vs 4 个 baselines

**保留原因**: 作为架构设计参考，说明如何扩展框架（如果将来需要）

<details>
<summary>原设计（仅供参考，不实施）</summary>

**目标**: 添加DQN改进版本

**步骤**:
1. Double DQN (1天)
2. Dueling DQN (1天)
3. D3QN (1天)

**时间**: 3天
</details>

### ~~Phase 4: A2C实现 (可选)~~ ❌ NOT IN CURRENT PROJECT SCOPE

<details>
<summary>原设计（仅供参考，不实施）</summary>

**时间**: 3-5天
</details>

---

## ⚠️ 设计权衡

### 为什么允许不同的update()签名？

**决策**: 不强制统一update签名，允许算法特定参数

**理由**:
- DQN需要: `update(batch)` - 从replay buffer批量更新
- PPO需要: `update(trajectory)` - 完整轨迹更新
- 强制统一会导致接口不自然

**替代方案** (rejected):
```python
# ❌ 方案1: 强制统一签名
def update(self, data: Dict) -> float:
    # 需要判断data格式，不直观

# ❌ 方案2: 多个update方法
def update_batch(self, batch): ...
def update_trajectory(self, traj): ...
# 接口复杂化
```

**采用方案** (chosen):
```python
# ✅ 方案3: 灵活签名
def update(self, *args, **kwargs) -> float:
    # 子类根据需要定义具体签名
    # 文档清楚说明每个算法的签名
```

### 为什么不用抽象Factory Pattern？

**决策**: 使用简单的ALGORITHM_REGISTRY字典

**理由**:
- Registry更直观，易于添加新算法
- 不需要复杂的工厂类层次
- 配置驱动，易于扩展

---

## ✅ 成功标准

### Phase 1完成标准
- [ ] BaseAgent接口定义完成
- [ ] OffPolicyTrainer实现完成
- [ ] DQNAgent继承BaseAgent
- [ ] `train.py --algorithm dqn` 可运行
- [ ] 训练结果与旧版一致

### Phase 2完成标准
- [ ] OnPolicyTrainer实现完成
- [ ] PPOAgent实现完成
- [ ] PPO训练收敛
- [ ] Level 1对比实验（DQN vs PPO）完成

### 最终验收标准
- [ ] 至少支持4种算法（DQN, Double DQN, PPO, A2C）
- [ ] 统一的`train.py`入口
- [ ] 所有算法Level 1测试通过
- [ ] 文档完整（每个算法有说明）
- [ ] 单元测试覆盖核心组件

---

**Date**: 2025-10-19
**Status**: 设计完成，待实施
**Next**: 开始Phase 1 - DQN重构
