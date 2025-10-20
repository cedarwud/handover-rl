#!/usr/bin/env python3
"""
Base Handover Environment - Gym-Compatible RL Environment

Gymnasium-compatible environment for satellite handover decision-making.

State Space (12 dimensions):
    - Signal Quality (3): RSRP, RSRQ, RS-SINR
    - Physical Parameters (7): Distance, Elevation, Doppler, Velocity,
                               Atmospheric Loss, Path Loss, Delay
    - 3GPP Offsets (2): Offset MO, Cell Offset

Action Space (2 discrete actions):
    - 0: Maintain (keep current serving satellite)
    - 1: Handover (switch to best neighbor satellite)

Reward Function:
    reward = w1 × QoS_improvement
           - w2 × handover_penalty
           + w3 × signal_quality
           - w4 × ping_pong_penalty

SOURCE:
- Gymnasium API: https://gymnasium.farama.org/
- 3GPP TS 38.331 v18.5.1: Handover procedures
- 3GPP TS 38.215 v18.1.0: Signal quality measurements
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class BaseHandoverEnvironment(gym.Env):
    """
    Base Handover Environment (Gymnasium-compatible).

    This is the base class for handover decision environments.
    Concrete implementations should inherit from this class.

    Attributes:
        observation_space: Box(12,) - 12-dimensional continuous state
        action_space: Discrete(2) - maintain or handover
    """

    # Gymnasium metadata
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 1
    }

    def __init__(self, config: Dict):
        """
        Initialize base handover environment.

        Args:
            config: Configuration dictionary with:
                - environment.state_dim: State space dimension (default: 12)
                - environment.action_dim: Action space dimension (default: 2)
                - environment.reward_weights: Reward function weights
                - environment.max_steps_per_episode: Maximum steps
        """
        super().__init__()

        self.config = config
        self.env_config = config.get('environment', {})

        # State and action dimensions
        self.state_dim = self.env_config.get('state_dim', 12)
        self.action_dim = self.env_config.get('action_dim', 2)

        # Define observation space: Box(12,)
        # [rsrp, rsrq, sinr, distance, elevation, doppler, velocity,
        #  atmospheric_loss, path_loss, propagation_delay, offset_mo, cell_offset]
        self.observation_space = spaces.Box(
            low=np.array([
                -150.0,   # rsrp_dbm (SOURCE: 3GPP TS 38.215 v18.1.0 Table 5.1.1-1)
                -30.0,    # rsrq_db (SOURCE: 3GPP TS 38.215 v18.1.0 Table 5.1.3-1)
                -10.0,    # rs_sinr_db (SOURCE: 3GPP TS 38.215 v18.1.0 Table 5.1.5-1)
                500.0,    # distance_km (LEO orbit altitude range)
                0.0,      # elevation_deg (SOURCE: 3GPP TR 38.821 Section 6.1.2)
                -50000.0, # doppler_shift_hz (SOURCE: ITU-R M.1184 Annex 1)
                -8000.0,  # radial_velocity_ms (LEO orbital velocity)
                0.0,      # atmospheric_loss_db (SOURCE: ITU-R P.676-13)
                120.0,    # path_loss_db (SOURCE: 3GPP TR 38.811 Section 6.6.3)
                0.0,      # propagation_delay_ms
                -10.0,    # offset_mo_db (SOURCE: 3GPP TS 38.331 Section 5.5.4)
                -10.0     # cell_offset_db
            ], dtype=np.float32),
            high=np.array([
                -30.0,    # rsrp_dbm
                -3.0,     # rsrq_db
                30.0,     # rs_sinr_db
                3000.0,   # distance_km
                90.0,     # elevation_deg
                50000.0,  # doppler_shift_hz
                8000.0,   # radial_velocity_ms
                30.0,     # atmospheric_loss_db
                200.0,    # path_loss_db
                50.0,     # propagation_delay_ms
                10.0,     # offset_mo_db
                10.0      # cell_offset_db
            ], dtype=np.float32),
            dtype=np.float32
        )

        # Define action space: Discrete(2)
        # 0 = maintain, 1 = handover
        self.action_space = spaces.Discrete(self.action_dim)

        # Reward function weights
        reward_weights = self.env_config.get('reward_weights', {})
        self.w_qos = reward_weights.get('qos_improvement', 1.0)
        self.w_handover = reward_weights.get('handover_penalty', 0.5)
        self.w_signal = reward_weights.get('signal_quality', 0.3)
        self.w_ping_pong = reward_weights.get('ping_pong_penalty', 1.0)

        # Episode settings
        self.max_steps = self.env_config.get('max_steps_per_episode', 1500)

        # Environment state
        self.current_step = 0
        self.total_handovers = 0
        self.last_handover_step = -1
        self.episode_rewards = []
        self.current_state = None

        # Statistics
        self.episode_count = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (environment-specific)

        Returns:
            observation: Initial state (12-dim numpy array)
            info: Additional information (dict)
        """
        # Set seed if provided
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)

        # Reset episode state
        self.current_step = 0
        self.total_handovers = 0
        self.last_handover_step = -1
        self.episode_rewards = []
        self.episode_count += 1

        # Get initial observation (to be implemented by subclass)
        observation = self._get_initial_observation()
        self.current_state = observation

        # Info dictionary
        info = {
            'episode': self.episode_count,
            'step': self.current_step
        }

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (0=maintain, 1=handover)

        Returns:
            observation: Next state (12-dim numpy array)
            reward: Reward for this step (float)
            terminated: Whether episode ended naturally (bool)
            truncated: Whether episode was truncated (timeout, error) (bool)
            info: Additional information (dict)
        """
        # Validate action
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Execute action and get next state
        next_state = self._execute_action(action)

        # Calculate reward
        reward = self._calculate_reward(action, self.current_state, next_state)
        self.episode_rewards.append(reward)

        # Update handover tracking
        if action == 1:  # handover
            self.total_handovers += 1
            self.last_handover_step = self.current_step

        # Update step counter
        self.current_step += 1

        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps

        # Update current state
        self.current_state = next_state

        # Info dictionary
        info = {
            'step': self.current_step,
            'total_handovers': self.total_handovers,
            'action_taken': action,
            'episode_reward': sum(self.episode_rewards)
        }

        return next_state, reward, terminated, truncated, info

    def _get_initial_observation(self) -> np.ndarray:
        """
        Get initial observation (to be implemented by subclass).

        Returns:
            observation: Initial 12-dim state
        """
        # Default: return zeros (subclass should override)
        return np.zeros(self.state_dim, dtype=np.float32)

    def _execute_action(self, action: int) -> np.ndarray:
        """
        Execute action and get next state (to be implemented by subclass).

        Args:
            action: Action to execute

        Returns:
            next_state: Next 12-dim state
        """
        # Default: return current state (subclass should override)
        return self.current_state

    def _calculate_reward(self, action: int, state: np.ndarray, next_state: np.ndarray) -> float:
        """
        Calculate reward for the transition.

        Reward = w1 × QoS_improvement
               - w2 × handover_penalty
               + w3 × signal_quality
               - w4 × ping_pong_penalty

        Args:
            action: Action taken
            state: Current state
            next_state: Next state

        Returns:
            reward: Calculated reward
        """
        # Component 1: QoS improvement (RSRP change)
        # RSRP is first dimension of state
        qos_improvement = 0.0
        if action == 1:  # handover
            current_rsrp = state[0]
            next_rsrp = next_state[0]
            rsrp_diff = next_rsrp - current_rsrp
            # Normalize to [-1, 1] (assume max difference ±60 dB)
            qos_improvement = np.clip(rsrp_diff / 60.0, -1.0, 1.0)

        # Component 2: Handover penalty
        handover_penalty = 1.0 if action == 1 else 0.0

        # Component 3: Signal quality (based on RSRP threshold)
        # SOURCE: 3GPP TS 38.133 v18.3.0 Table 10.1.19.2-1
        current_rsrp = state[0]
        if current_rsrp > -90:  # Good signal
            signal_quality = 0.5
        elif current_rsrp < -110:  # Poor signal
            signal_quality = -0.5
        else:
            signal_quality = 0.0

        # Component 4: Ping-pong penalty (rapid consecutive handovers)
        # SOURCE: 3GPP TS 36.839 v11.1.0 Section 6.2.3.2
        ping_pong_penalty = 0.0
        if action == 1 and self.last_handover_step >= 0:
            steps_since_last = self.current_step - self.last_handover_step
            if steps_since_last < 10:  # < 50s at 5s interval
                ping_pong_penalty = 1.0

        # Total reward
        total_reward = (
            self.w_qos * qos_improvement
            - self.w_handover * handover_penalty
            + self.w_signal * signal_quality
            - self.w_ping_pong * ping_pong_penalty
        )

        return total_reward

    def _is_terminated(self) -> bool:
        """
        Check if episode should terminate naturally.

        Returns:
            terminated: Whether episode ended
        """
        # Default: never terminate naturally (subclass can override)
        # Episodes typically end by truncation (max steps)
        return False

    def render(self):
        """
        Render the environment.

        For handover environment, this could display:
        - Current satellite information
        - Signal quality metrics
        - Handover history
        """
        if self.current_state is None:
            return

        print(f"Step {self.current_step}:")
        print(f"  RSRP: {self.current_state[0]:.2f} dBm")
        print(f"  RSRQ: {self.current_state[1]:.2f} dB")
        print(f"  SINR: {self.current_state[2]:.2f} dB")
        print(f"  Distance: {self.current_state[3]:.2f} km")
        print(f"  Elevation: {self.current_state[4]:.2f}°")
        print(f"  Total Handovers: {self.total_handovers}")

    def close(self):
        """Close the environment and clean up resources."""
        pass

    def get_episode_statistics(self) -> Dict:
        """
        Get statistics for the current episode.

        Returns:
            stats: Dictionary with episode statistics
        """
        return {
            'episode': self.episode_count,
            'total_steps': self.current_step,
            'total_handovers': self.total_handovers,
            'total_reward': sum(self.episode_rewards),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'handover_rate': self.total_handovers / max(self.current_step, 1)
        }


# Example usage
if __name__ == "__main__":
    print("BaseHandoverEnvironment - Gym-Compatible Interface")
    print("=" * 60)

    # Example configuration
    config = {
        'environment': {
            'state_dim': 12,
            'action_dim': 2,
            'reward_weights': {
                'qos_improvement': 1.0,
                'handover_penalty': 0.5,
                'signal_quality': 0.3,
                'ping_pong_penalty': 1.0
            },
            'max_steps_per_episode': 100
        }
    }

    # Create environment
    env = BaseHandoverEnvironment(config)

    print(f"✅ Observation space: {env.observation_space}")
    print(f"✅ Action space: {env.action_space}")

    # Reset environment
    state, info = env.reset()
    print(f"✅ Initial state shape: {state.shape}")

    # Take a few random steps
    print("\nTaking 5 random steps:")
    for i in range(5):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: action={action}, reward={reward:.3f}, handovers={info['total_handovers']}")

        if terminated or truncated:
            break

    # Get episode statistics
    stats = env.get_episode_statistics()
    print(f"\n✅ Episode statistics: {stats}")

    print("\n" + "=" * 60)
    print("BaseHandoverEnvironment verified!")
