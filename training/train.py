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
from wrappers import SkipFrameWrapper, FrameDownsampleWrapper
from gym.wrappers import FrameStack
from model import DQNAgent

def create_env():
    """
    Create the environment for the Mario game.
    """
    # Create the environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrameWrapper(env, skip=4)
    env = FrameDownsampleWrapper(env)
    env = FrameStack(env, num_stack=4)
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    print("state size:", state_size)
    print("action size:", action_size)
    return env, state_size, action_size

def save_checkpoint(agent, episode):
    
    """
    Save the model checkpoint.
    """
    checkpoint_dir = "mario_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{episode}.pth")
    torch.save(agent.model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def train_agent(agent, env, hyperparameters):
    
    """
    Train the agent using the DQN algorithm.
    """
    num_episodes = hyperparameters["num_episodes"]
    epsilon = hyperparameters["epsilon_start"]
    epsilon_end = hyperparameters["epsilon_end"]
    epsilon_decay = hyperparameters["epsilon_decay"]

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:

            if random.random() < epsilon:
                action = randrange(agent.action_size)
            else:
                action = agent.get_action(state)

            next_state, reward, done, _ = env.step(action)

            total_reward += reward
            agent.replay_buffer.add((state, action, reward, next_state, done))
            state = next_state

            if agent.replay_buffer.size > agent.batch_size:
                agent.train()

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        beta = 0.4
        beta_increment = (1.0 - beta) / num_episodes
        agent.replay_buffer.beta = min(1.0, agent.replay_buffer.beta + beta_increment)
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
        if (episode + 1) % 100 == 0:
            save_checkpoint(agent, episode + 1)

if __name__ == "__main__":
    
    agent_hyperparameters = {
        "gamma": 0.99,
        "learning_rate": 0.001,
        "batch_size": 32,
        "update_target_every": 1000,
        "replay_buffer_hyperparameters": {
            "beta_start": 0.4,
            "alpha": 0.6,
            "epsilon": 1e-6,
            "buffer_size": 1000000,
        }
    }
    train_hyperparameters = {
        "num_episodes": 1000,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
    }

    env, state_size, action_size = create_env()
    agent = DQNAgent(state_size, action_size, agent_hyperparameters)
    train_agent(agent, env, train_hyperparameters)