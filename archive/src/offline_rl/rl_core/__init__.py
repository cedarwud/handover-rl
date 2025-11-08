"""
RL Core Framework

Algorithm-agnostic base classes for reinforcement learning.

Classes:
    BaseRLAgent: Abstract base class for all RL agents
    BaseHandoverEnvironment: Gymnasium-compatible handover environment
    UniversalRLTrainer: Algorithm-agnostic training loop

Usage:
    from src.rl_core import BaseRLAgent, BaseHandoverEnvironment, UniversalRLTrainer

    # Implement custom agent
    class MyAgent(BaseRLAgent):
        def select_action(self, state, eval_mode=False):
            ...
        def update(self):
            ...

    # Create environment and trainer
    env = MyEnvironment(config)  # Inherits from BaseHandoverEnvironment
    agent = MyAgent(state_dim=12, action_dim=2, config)
    trainer = UniversalRLTrainer(agent, env, config=config)
    trainer.train()
"""

from .base_agent import BaseRLAgent, DummyAgent
from .base_environment import BaseHandoverEnvironment
from .universal_trainer import UniversalRLTrainer

__all__ = [
    'BaseRLAgent',
    'DummyAgent',
    'BaseHandoverEnvironment',
    'UniversalRLTrainer'
]

__version__ = "2.0.0"
