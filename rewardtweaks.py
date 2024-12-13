import gymnasium as gym
from gymnasium.wrappers import RewardWrapper

class AlternateLegRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_leg = None
        self.last_hip_positions = None

    def reward(self, reward):
        # Custom reward logic
        current_hip_positions = self.env.unwrapped.hull.position if hasattr(self.env.unwrapped, "hull") else [0, 0]
        if self.last_hip_positions is None:
            self.last_hip_positions = current_hip_positions
        left_leg = current_hip_positions[0]
        right_leg = current_hip_positions[1]
        leading_leg = "left" if left_leg > right_leg else "right"
        if self.last_leg is not None:
            if self.last_leg != leading_leg:
                reward += 5.0
            else:
                reward -= 1.0
        self.last_leg = leading_leg
        self.last_hip_positions = current_hip_positions
        return reward
