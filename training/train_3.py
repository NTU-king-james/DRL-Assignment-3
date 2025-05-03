import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import matplotlib.pyplot as plt
import torch
import random, datetime, os
from pathlib import Path
import torch.nn as nn
import numpy as np
from random import randrange
from wrappers import SkipFrameWrapper, FrameDownsampleWrapper
from gym.wrappers import FrameStack, TimeLimit
from tensordict import TensorDict
from torchrl.data import MultiStep
from torchrl.data.replay_buffers import PrioritizedSampler
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from logger import MetricLogger

# Current: Duel + double + PER + Noisy nets
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
    env = TimeLimit(env, max_episode_steps=3000)
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    print("state size:", state_size)
    print("action size:", action_size)
    return env, state_size, action_size

import torch
import torch.nn as nn

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        def scale_noise(size):
            # μ=0、 σ=1
            x = torch.randn(size)
            # 開根號，保留正負值，更平滑
            return x.sign().mul(x.abs().sqrt())

        eps_in = scale_noise(self.in_features)
        eps_out = scale_noise(self.out_features)
        # * means 外積
        self.weight_epsilon.copy_(eps_out.unsqueeze(1) * eps_in.unsqueeze(0))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

class NoisyDuelNet(nn.Module):
    """Dueling DQN + NoisyNet for 84x84 inputs"""
    def __init__(self, input_dim, output_dim, sigma_init=0.017):
        super().__init__()
        c, h, w = input_dim
        if h != 84 or w != 84:
            raise ValueError(f"Expecting 84x84 input, got: {h}x{w}")

        self.online = self._build_dueling_net(c, output_dim, sigma_init)
        self.target = self._build_dueling_net(c, output_dim, sigma_init)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x, model="online"):
        net = self.online if model == "online" else self.target
        return net(x)

    def reset_noise(self):
        """Reset noise in all NoisyLinear layers for fresh exploration"""
        for m in self.online.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def _build_dueling_net(self, in_channels, output_dim, sigma_init):
        # shared CNN features
        feature = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),        nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),        nn.ReLU(),
            nn.Flatten()
        )
        # determine flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            n_flatten = feature(dummy).shape[1]

        # value and advantage streams with NoisyLinear
        value = nn.Sequential(
            NoisyLinear(n_flatten, 512, sigma_init), nn.ReLU(),
            NoisyLinear(512, 1, sigma_init)
        )
        advantage = nn.Sequential(
            NoisyLinear(n_flatten, 512, sigma_init), nn.ReLU(),
            NoisyLinear(512, output_dim, sigma_init)
        )

        class NoisyDuelingBranch(nn.Module):
            def __init__(self, feat, val, adv):
                super().__init__()
                self.feat = feat
                self.val  = val
                self.adv  = adv

            def forward(self, x):
                x = self.feat(x)
                v = self.val(x)                   # [B,1]
                a = self.adv(x)                   # [B,output_dim]
                return v + a - a.mean(dim=1, keepdim=True)

        return NoisyDuelingBranch(feature, value, advantage)

        
class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        self.batch_size = 32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = NoisyDuelNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)
        self.curr_step = 0

        self.save_every = 5e5  # no. of experiences between saving Mario Net

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(``LazyFrame``): A single observation of the current state, dimension is (state_dim)
        Outputs:
        ``action_idx`` (``int``): An integer representing which action Mario will perform
        """

        # EXPLOIT
        self.net.train()
        self.net.reset_noise()

        state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        action_values = self.net(state, model="online")
        action_idx = torch.argmax(action_values, axis=1).item()

        # increment step
        self.curr_step += 1
        return action_idx

class Mario(Mario):  # subclassing for continuity
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.sampler = PrioritizedSampler(
            max_capacity=100000,
            alpha=0.6,
            beta=0.4,
            eps=1e-6,
        )
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")), 
                                             sampler=self.sampler,
                                             priority_key="td_error",
                                             batch_size=32)
        self.batch_size = 32
        self.max_priority = 1.0
        self.beta_end = 1.0
        self.anneal_steps = 1000000
        self.step_count = 0
        self.beta_start = 0.4

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (``LazyFrame``),
        next_state (``LazyFrame``),
        action (``int``),
        reward (``float``),
        done(``bool``)),
        td_error (``float``): TD error for the experience
        """
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])
        # self.memory.append((state, next_state, action, reward, done,))
        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action,
                                     "reward": reward, "done": done, "td_error": self.max_priority}, batch_size=[]))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        self.step_count += 1
        if hasattr(self.sampler, "_beta"):
            frac = min(1.0, self.step_count / self.anneal_steps)
            self.sampler._beta= self.beta_start + frac * (self.beta_end - self.beta_start)
        else:
            raise NameError("No _beta in sampler")
       

        batch, info = self.memory.sample(return_info=True)
        batch = batch.to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        is_weights = info["_weight"].to(self.device)
        index = info["index"]
        if is_weights is None or index is None:
            print(info.keys())
            raise ValueError("IS weights and indices are None. Check your memory sampling.")
        
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze(), is_weights, index
    
    def update_priority(self, indices, td_errors):
        """
        Update the TD error of the experiences in memory
        """
        td_errors = np.abs(td_errors)
        self.memory.update_priority(indices, td_errors)
        self.max_priority = max(self.max_priority, np.max(td_errors))
    
class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.gamma = 0.9

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss(reduction="none")

    def update_Q_online(self, td_estimate, td_target, is_weight):
        loss_per_sample = self.loss_fn(td_estimate, td_target)
        weight_loss = (loss_per_sample * is_weight).mean()
        self.optimizer.zero_grad()
        weight_loss.backward()
        self.optimizer.step()
        return weight_loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

class Mario(Mario):
    def save(self, e):
        save_path = (
            self.save_dir / f"mario_net_episode_{e}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict()),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at episode {e}")

class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done, is_weight, indices = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)
        td_error = td_est - td_tgt
        td_error = td_error.detach().cpu().numpy()
        # Update priority
        self.update_priority(indices, td_error)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt, is_weight)

        return (td_est.mean().item(), loss)


if __name__ == "__main__":
    # Create the environment
    env, state_size, action_size = create_env()
    
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()

    save_dir = Path("checkpoints_noisy") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

    logger = MetricLogger(save_dir)

    episodes = 10000
    for e in range(episodes):

        state = env.reset()

        # Play the game!
        while True:

            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, info = env.step(action)
            truncated = info.get("TimeLimit.truncated", False)
            done = done or truncated
            
            # Remember
            mario.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = mario.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            # Check if end of game
            if done or info["flag_get"]:
                break

        logger.log_episode()

        if (e % 10 == 0) or (e == episodes - 1):
            logger.record(episode=e, epsilon=0, step=mario.curr_step)
        if (e + 1) % 500 == 0:
            mario.save(e + 1)