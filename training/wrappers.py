import gym
import numpy as np
from collections import deque
import cv2
from gym.spaces import Box
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

class MaxAndSkipFrameWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        # 用來暫存最近 skip 幀的觀測
        self._obs_buffer = deque(maxlen=skip)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # 每次重置時，把第一幀重複填滿 buffer
        self._obs_buffer.clear()
        for _ in range(self._skip):
            self._obs_buffer.append(obs)
        return obs

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        # 重複執行 skip 次 action
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        # 把 buffer 裡的幀做 pixel-wise max 合併, 留下 value 最大的幀
        max_frame = np.max(np.stack(self._obs_buffer, axis=0), axis=0)
        return max_frame, total_reward, done, info

class FrameDownsampleWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 重新設定 observation_space 為 84x84x1 的 uint8
        self.observation_space = Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, obs):
        # 1. 轉灰階
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # 2. 縮放到 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        # 3. 增加 channel 維度
        return resized[:, :, None]

class ImageToPyTorchWrapper(gym.ObservationWrapper):
    """
    Convert images from HxWxC to CxHxW for PyTorch.
    """
    def __init__(self, env):
        super().__init__(env)
        # Original observation_space: (H, W, C)
        h, w, c = self.observation_space.shape
        # New observation_space: (C, H, W)
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(c, h, w),
            dtype=self.observation_space.dtype
        )

    def observation(self, obs):
        # obs: (H, W, C) -> (C, H, W)
        return np.transpose(obs, (2, 0, 1))


class FrameBufferWrapper(gym.ObservationWrapper):
    """
    Stack the last k frames along the channel dimension.
    """
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        # Expect underlying obs shape: (H, W, C)
        h, w, c = self.observation_space.shape
        self.frames = deque(maxlen=k)
        # New observation_space: (H, W, C * k)
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(h, w, c * k),
            dtype=self.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Fill deque with initial observation
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_observation()

    def observation(self, obs):
        # Append new frame and return stacked result
        self.frames.append(obs)
        return self._get_observation()

    def _get_observation(self):
        # Stack along channel axis
        return np.concatenate(list(self.frames), axis=2)
    
class NormalizeFloats(gym.ObservationWrapper):
    """
    將 uint8 影像 [0,255] 正規化到 float32 [0.0,1.0]
    """
    def __init__(self, env):
        super().__init__(env)
        # 取用原本的 observation_space
        old_space = self.observation_space
        # 重新定義 observation_space 為 float32, 範圍 [0.0,1.0]
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=old_space.shape,
            dtype=np.float32
        )

    def observation(self, obs):
        # 轉成 float32 並除以 255
        return obs.astype(np.float32) / 255.0