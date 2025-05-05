import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import matplotlib.pyplot as plt
import random, datetime, os
from pathlib import Path
import numpy as np
import math
from random import randrange
from collections import deque
from wrappers import SkipFrameWrapper, FrameDownsampleWrapper
from gym.wrappers import FrameStack, TimeLimit
from tensordict import TensorDict
from logger import MetricLogger
import torch
import torch.nn as nn
import torch.nn.functional as F

# reference: https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
# Current: Duel + double + PER + Noisy nets + n step
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
        self.batch_size = 32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = NoisyDuelNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)
        self.curr_step = 0

        # self.save_every = 5e5  # no. of experiences between saving Mario Net

    def act(self, state):
        
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

# PER + 4-step
class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        
           
        self.alpha=0.6
        self.beta=0.4
        self.beta_increment = (1.0 - self.beta) / 2000000
        self.beta_end=1.0
        self.eps=1e-6
        
        self.memory = deque(maxlen=100000)
        self.priorities = deque(maxlen=100000)
        self.batch_size = 32

        self.max_priority = 1.0
        self.gamma = 0.9
        self.nstep_buffer = deque(maxlen=4)
        self.step_count = 0

    def cache(self, state, next_state, action, reward, done):
            
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.as_tensor(state, dtype=torch.float32, device='cpu')
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device='cpu')
        
        action = torch.as_tensor([action], dtype=torch.int64, device='cpu')
        reward = torch.as_tensor([reward], dtype=torch.float32, device='cpu')
        done = torch.as_tensor([done], dtype=torch.bool, device='cpu')
        self.nstep_buffer.append(TensorDict({
            "state": state, "next_state": next_state,
            "action": action, "reward": reward, "done": done
        }, batch_size=[]))

        if len(self.nstep_buffer) == 4:
            G = 0
            for idx, td in enumerate(self.nstep_buffer):
                G += self.gamma**idx * td["reward"].item()

            first = self.nstep_buffer[0].clone()
            first["reward"] = torch.as_tensor([G], dtype=torch.float32, device='cpu')
            first["next_state"] = self.nstep_buffer[-1]["next_state"]
            first["done"] = self.nstep_buffer[-1]["done"]

            self.memory.append(first)
            self.priorities.append(self.max_priority)
            self.nstep_buffer.popleft()

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        self.step_count += 1
        self.beta = min(1.0, self.beta + self.beta_increment)
        prios = np.array(self.priorities, dtype=np.float32)
        scaled = prios ** self.alpha
        probs = scaled / scaled.sum()
        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        is_weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        is_weights = torch.tensor(is_weights / is_weights.max(), dtype=torch.float32, device=self.device)
        batch = [self.memory[idx] for idx in indices]
        batch = TensorDict.stack(batch, dim=0)
        batch = batch.to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))

        
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze(), is_weights, indices
    
    def update_priority(self, indices, td_errors):
        """
        Update the TD error of the experiences in memory
        """
        td_errors = np.abs(td_errors)
        for idx, err in zip(indices, td_errors):
            new_prio = err + self.eps
            self.priorities[idx] = new_prio
            if new_prio > self.max_priority:
                self.max_priority = new_prio

    def reset_nstep_buffer(self):
        """
        Reset the replay buffer
        """
        self.nstep_buffer.clear()
    
class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)

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
        return (reward + (1 - done.float()) * self.gamma**4 * next_Q).float()

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
        state, next_state, action, reward, done, is_weights, indices = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)
        td_error = td_est - td_tgt
        td_error = td_error.detach().cpu().numpy()
        # Update priority
        self.update_priority(indices, td_error)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt, is_weights)

        return (td_est.mean().item(), loss)


if __name__ == "__main__":
    # Create the environment
    env, state_size, action_size = create_env()
    
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()

    save_dir = Path("checkpoints_rainbow") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

    logger = MetricLogger(save_dir)

    episodes = 10000
    for e in range(episodes):

        state = env.reset()
        mario.reset_nstep_buffer()
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
        if (e + 1) % 500 == 0:
            save_path = (
                save_dir / f"mario_net_episode_{e+1}.chkpt"
            )
            torch.save(
                dict(model=mario.net.state_dict()),
                save_path,
            )
            print(f"MarioNet saved to {save_path} at episode {e}")

        if (e % 10 == 0) or (e == episodes - 1):
            logger.record(episode=e, epsilon=0, step=mario.curr_step)