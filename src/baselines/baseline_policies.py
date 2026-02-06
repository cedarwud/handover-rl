"""
Baseline Policies for LEO Satellite Handover

Implements simple heuristic policies for comparison with DQN:
1. Random: Random satellite selection
2. Always Stay: Never switch satellites
3. Max RSRP: Always choose highest RSRP satellite
4. Max Elevation: Always choose highest elevation satellite
5. Max RVT: Always choose satellite with maximum Remaining Visible Time (greedy)

These baselines help establish the performance improvement of the learned DQN policy.

Reference:
- IEEE TAES 2024: "User-Centric Satellite Handover for Multiple Traffic Profiles Using Deep Q-Learning"
- Baselines are standard comparison methods in RL-LEO research

Last updated: 2025-12-18
"""

import numpy as np
from typing import Tuple, Optional


class BaselinePolicy:
    """Base class for baseline policies"""

    def __init__(self, name: str):
        self.name = name

    def select_action(self, observation: np.ndarray, info: dict = None) -> int:
        """
        Select action based on policy

        Args:
            observation: State observation (K, 14) array
            info: Optional additional information

        Returns:
            action: Integer action (0 = stay, 1-K = switch to candidate)
        """
        raise NotImplementedError

    def reset(self):
        """Reset policy state (if needed)"""
        pass


class RandomPolicy(BaselinePolicy):
    """Random policy: uniformly random satellite selection"""

    def __init__(self):
        super().__init__("Random")

    def select_action(self, observation: np.ndarray, info: dict = None) -> int:
        num_actions = observation.shape[0] + 1  # K candidates + stay
        return np.random.randint(0, num_actions)


class AlwaysStayPolicy(BaselinePolicy):
    """Always Stay: never perform handover"""

    def __init__(self):
        super().__init__("Always Stay")

    def select_action(self, observation: np.ndarray, info: dict = None) -> int:
        return 0  # Always stay


class MaxRSRPPolicy(BaselinePolicy):
    """Max RSRP: always choose satellite with highest RSRP"""

    def __init__(self):
        super().__init__("Max RSRP")

    def select_action(self, observation: np.ndarray, info: dict = None) -> int:
        # RSRP is at index 10 in state vector
        rsrp_values = observation[:, 10]

        # Find max RSRP
        max_idx = np.argmax(rsrp_values)

        # Action 0 = stay (current satellite, first in list)
        # Action i+1 = switch to candidate i
        # If max is current satellite (index 0), stay
        if max_idx == 0:
            return 0
        else:
            return max_idx


class MaxElevationPolicy(BaselinePolicy):
    """Max Elevation: always choose satellite with highest elevation angle"""

    def __init__(self):
        super().__init__("Max Elevation")

    def select_action(self, observation: np.ndarray, info: dict = None) -> int:
        # Elevation is at index 0 in state vector
        elevation_values = observation[:, 0]

        # Find max elevation
        max_idx = np.argmax(elevation_values)

        # If max is current satellite, stay
        if max_idx == 0:
            return 0
        else:
            return max_idx


class MaxRVTPolicy(BaselinePolicy):
    """Max RVT (Greedy): always choose satellite with maximum Remaining Visible Time

    This is a greedy policy that maximizes immediate RVT without considering
    long-term consequences. Should be better than random but worse than DQN.
    """

    def __init__(self):
        super().__init__("Max RVT (Greedy)")

    def select_action(self, observation: np.ndarray, info: dict = None) -> int:
        # RVT is at index 13 (last dimension) in state vector
        rvt_values = observation[:, 13]

        # Find max RVT
        max_idx = np.argmax(rvt_values)

        # If max is current satellite, stay
        if max_idx == 0:
            return 0
        else:
            return max_idx


class RoundRobinPolicy(BaselinePolicy):
    """Round Robin: cycle through satellites in order

    Simple deterministic policy that switches to next satellite periodically.
    """

    def __init__(self, switch_interval: int = 12):
        """
        Args:
            switch_interval: Number of steps before switching (default: 12 = 1 min)
        """
        super().__init__("Round Robin")
        self.switch_interval = switch_interval
        self.steps_since_switch = 0
        self.current_candidate_idx = 0

    def select_action(self, observation: np.ndarray, info: dict = None) -> int:
        num_candidates = observation.shape[0] - 1  # Exclude current

        if self.steps_since_switch >= self.switch_interval:
            # Time to switch
            self.current_candidate_idx = (self.current_candidate_idx + 1) % num_candidates
            self.steps_since_switch = 0
            return self.current_candidate_idx + 1  # +1 because 0 is stay
        else:
            # Stay
            self.steps_since_switch += 1
            return 0

    def reset(self):
        self.steps_since_switch = 0
        self.current_candidate_idx = 0


# Dictionary of all baseline policies
BASELINE_POLICIES = {
    'random': RandomPolicy(),
    'always_stay': AlwaysStayPolicy(),
    'max_rsrp': MaxRSRPPolicy(),
    'max_elevation': MaxElevationPolicy(),
    'max_rvt': MaxRVTPolicy(),
}


def get_baseline_policy(name: str) -> BaselinePolicy:
    """Get baseline policy by name"""
    if name not in BASELINE_POLICIES:
        raise ValueError(f"Unknown baseline policy: {name}. "
                        f"Available: {list(BASELINE_POLICIES.keys())}")
    return BASELINE_POLICIES[name]
