import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import os
import matplotlib.pyplot as plt
import torch
import random
from collections import deque
import torch.nn as nn
import numpy as np
from random import randrange
from wrappers import MaxAndSkipFrameWrapper, FrameDownsampleWrapper, ImageToPyTorchWrapper, FrameBufferWrapper, NormalizeFloats

def create_env():
    """
    Create the environment for the Mario game.
    """
    # Create the environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    return env
    env = MaxAndSkipFrameWrapper(env, skip=4)
    env = FrameDownsampleWrapper(env)
    env = ImageToPyTorchWrapper(env)
    env = FrameBufferWrapper(env)
    env = NormalizeFloats(env)
    return env

class DQNModel(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNModel, self).__init__()
        self._input_shape = input_shape
        self._num_actions = num_actions
 
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
 
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
 
    def forward(self, x):
        # x.size()[0] = batch size
        x = self.features(x).view(x.size()[0], -1)
        return self.fc(x)
 
    def act(self, state, epsilon, device):
        if random() > epsilon:
            state = torch.FloatTensor(np.float32(state)).unsqueeze(0).to(device)
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = randrange(self._num_actions)
        return action
    
if __name__ == "__main__":
    env = create_env()

    # Play randomly
    done = False
    # reset the environment and capture the initial state
    reset_result = env.reset()
    step = 0
    total_reward = 0
    while not done:
        action = randrange(len(COMPLEX_MOVEMENT))
        result = env.step(action)
        # handle old vs new step API
        if len(result) == 5:
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            next_state, reward, done, info = result
        total_reward += reward
        env.render()  # display the game window in real time
        #print("reward: ", reward)
        #env.render()
        state = next_state
        step += 1
    print("Episode finished after {} timesteps".format(step))
    print("Final reward: ", total_reward)

    env.close()