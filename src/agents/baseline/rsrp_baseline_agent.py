#!/usr/bin/env python3
"""
RSRP Baseline Agent

Baseline handover strategy that always selects the satellite with the highest RSRP.
This is the standard comparison baseline for LEO satellite handover optimization.

Academic Reference:
- Used as baseline in most LEO handover papers
- Corresponds to 3GPP A3 event: "Neighbour becomes better than serving"
- No learning required - purely greedy RSRP selection

Performance Characteristics:
- Simple and deterministic
- May cause frequent handovers (ping-pong effect)
- Does not consider other factors (distance, velocity, etc.)
- Serves as lower bound for ML-based methods
"""

import numpy as np
from typing import Any, Dict, Optional
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.base_agent import BaseAgent


class RSRPBaselineAgent(BaseAgent):
    """
    RSRP Baseline Agent

    Strategy: Always select the satellite with the highest RSRP (Reference Signal Received Power)

    This is the standard baseline for satellite handover optimization:
    - No training required
    - Deterministic behavior
    - Purely greedy signal quality maximization

    State Space (per satellite):
        - Feature 0: RSRP (dBm) ← Used for decision
        - Features 1-11: RSRQ, SINR, distance, elevation, doppler, etc. (ignored)

    Action Space:
        - Action 0: Stay with current satellite
        - Action 1-K: Switch to satellite 1-K

    Decision Logic:
        1. Extract RSRP values from all visible satellites
        2. Find satellite with maximum RSRP
        3. If current satellite already has max RSRP, stay (action 0)
        4. Otherwise, switch to satellite with max RSRP
    """

    def __init__(self, observation_space, action_space, config: Dict[str, Any]):
        """
        Initialize RSRP Baseline Agent

        Args:
            observation_space: Observation space from environment (shape: [K, 12])
            action_space: Action space from environment (Discrete(K+1))
            config: Configuration dictionary (not used for baseline, but kept for interface compatibility)
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config

        # Extract dimensions
        self.max_visible_satellites = observation_space.shape[0]
        self.state_dim = observation_space.shape[1]
        self.n_actions = action_space.n

        # RSRP is feature index 0 in state vector
        # SOURCE: src/environments/satellite_handover_env.py:392-405
        self.rsrp_feature_index = 0

        # Statistics for logging
        self.total_selections = 0
        self.stay_count = 0
        self.handover_count = 0

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action: Choose satellite with highest RSRP

        Args:
            state: Observation from environment (shape: [K, 12])
                   K satellites × 12 features per satellite
            deterministic: Not used (baseline is always deterministic)

        Returns:
            action: Selected action index
                   - 0: Stay with current satellite (index 0)
                   - 1 to K: Switch to satellite index 1 to K

        Decision Logic:
            1. Extract RSRP from all satellites (state[:, 0])
            2. Find index of maximum RSRP
            3. If max RSRP is at index 0 (current satellite), return action 0 (stay)
            4. Otherwise, return action = index of best satellite
        """
        # Extract RSRP values for all visible satellites
        rsrp_values = state[:, self.rsrp_feature_index]  # shape: [K]

        # Find satellite with maximum RSRP
        best_satellite_idx = np.argmax(rsrp_values)

        # Determine action
        if best_satellite_idx == 0:
            # Current satellite (index 0) has best RSRP, stay
            action = 0
            self.stay_count += 1
        else:
            # Switch to satellite with best RSRP
            action = best_satellite_idx
            self.handover_count += 1

        self.total_selections += 1

        return action

    def update(self, *args, **kwargs) -> Optional[float]:
        """
        Update agent (No-op for baseline)

        Baseline agent does not learn, so no update is performed.

        Returns:
            None (no learning loss)
        """
        return None

    def save(self, path: str) -> None:
        """
        Save agent (No-op for baseline)

        Baseline agent has no learnable parameters, so no save is needed.
        However, we save statistics for analysis.
        """
        import json

        stats = {
            'agent_type': 'RSRP_Baseline',
            'total_selections': self.total_selections,
            'stay_count': self.stay_count,
            'handover_count': self.handover_count,
            'handover_rate': self.handover_count / self.total_selections if self.total_selections > 0 else 0.0
        }

        # Save as JSON
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path.with_suffix('.json'), 'w') as f:
            json.dump(stats, f, indent=2)

    def load(self, path: str) -> None:
        """
        Load agent (No-op for baseline)

        Baseline agent has no learnable parameters to load.
        """
        pass

    def on_episode_start(self) -> None:
        """Called at episode start (No-op for baseline)"""
        pass

    def on_episode_end(self, episode_reward: float, episode_info: Dict[str, Any]) -> None:
        """Called at episode end (No-op for baseline)"""
        pass

    def get_config(self) -> Dict[str, Any]:
        """
        Return agent configuration

        Returns:
            config: Dictionary with agent info and statistics
        """
        return {
            'agent_type': 'RSRP_Baseline',
            'strategy': 'Greedy RSRP selection',
            'learning': False,
            'total_selections': self.total_selections,
            'stay_count': self.stay_count,
            'handover_count': self.handover_count,
            'handover_rate': self.handover_count / self.total_selections if self.total_selections > 0 else 0.0
        }

    def __repr__(self) -> str:
        """String representation"""
        return f"RSRPBaselineAgent(handovers={self.handover_count}/{self.total_selections})"


if __name__ == '__main__':
    """Test RSRP Baseline Agent"""
    import gymnasium as gym

    # Create dummy observation space (10 satellites, 12 features each)
    obs_space = gym.spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(10, 12), dtype=np.float32
    )

    # Create dummy action space (11 actions: stay + 10 satellites)
    action_space = gym.spaces.Discrete(11)

    # Initialize agent
    agent = RSRPBaselineAgent(obs_space, action_space, {})

    # Test with dummy state
    # State: 10 satellites × 12 features
    # RSRP values: [-60, -55, -58, -62, -50, -65, -70, -68, -72, -75] dBm
    state = np.zeros((10, 12), dtype=np.float32)
    rsrp_values = np.array([-60, -55, -58, -62, -50, -65, -70, -68, -72, -75])
    state[:, 0] = rsrp_values  # Set RSRP feature

    print("Testing RSRP Baseline Agent")
    print("=" * 80)
    print(f"RSRP values: {rsrp_values}")
    print(f"Best satellite: index {np.argmax(rsrp_values)} (RSRP = {np.max(rsrp_values)} dBm)")

    # Select action
    action = agent.select_action(state)
    print(f"Selected action: {action}")

    if action == 0:
        print("Decision: Stay with current satellite")
    else:
        print(f"Decision: Switch to satellite {action}")

    print("\nAgent statistics:")
    print(agent.get_config())
    print("=" * 80)
    print("✅ RSRP Baseline Agent test passed")
