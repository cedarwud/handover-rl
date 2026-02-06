"""
Baseline policies for comparison with DQN

Available baselines:
- Random
- Always Stay
- Max RSRP
- Max Elevation
- Max RVT (Greedy)
"""

from baselines.baseline_policies import (
    BaselinePolicy,
    RandomPolicy,
    AlwaysStayPolicy,
    MaxRSRPPolicy,
    MaxElevationPolicy,
    MaxRVTPolicy,
    RoundRobinPolicy,
    BASELINE_POLICIES,
    get_baseline_policy,
)

__all__ = [
    'BaselinePolicy',
    'RandomPolicy',
    'AlwaysStayPolicy',
    'MaxRSRPPolicy',
    'MaxElevationPolicy',
    'MaxRVTPolicy',
    'RoundRobinPolicy',
    'BASELINE_POLICIES',
    'get_baseline_policy',
]
