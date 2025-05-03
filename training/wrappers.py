import gym
import numpy as np
import cv2
from gym.spaces import Box
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


class FrameDownsampleWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 重新設定 observation_space 為 1x84x84 的 float32
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(84, 84), dtype=np.float32
        )

    def observation(self, obs):
        # 1. (240, 256, 3) to (240, 256, 1)
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # 2. (240, 256, 1) to (84, 84)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        normalize_obs = resized.astype(np.float32) / 255.0
        return normalize_obs
    
class SkipFrameWrapper(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

