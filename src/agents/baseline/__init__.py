"""
Baseline Agents Module

Collection of non-learning baseline strategies for comparison:
- RSRP Baseline: Always select satellite with highest RSRP
- Random Baseline: Random satellite selection
- Distance Baseline: Always select nearest satellite

These baselines serve as performance lower bounds for ML-based agents.
"""

from .rsrp_baseline_agent import RSRPBaselineAgent

__all__ = ['RSRPBaselineAgent']
