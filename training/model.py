import torch
import random
import torch.nn as nn
import numpy as np
from random import randrange
from collections import deque

class DuelNet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DuelNet, self).__init__()
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
 
        test_input = torch.zeros(1, *input_shape)
        test_output = self.features(test_input)
        feature_size = test_output.view(1, -1).size(1)
 
        self.value = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
 
    def forward(self, x):
        # x.size()[0] = batch size
        x = self.features(x).view(x.size()[0], -1)
        value = self.value(x)
        advantage = self.advantage(x)
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q

# Priority Replay Buffer
class ReplayBuffer:
    def __init__(self, hyperparameters):
        # TODO: Initialize the buffer
        self.capacity = hyperparameters["buffer_size"]
        self.alpha = hyperparameters["alpha"]
        self.beta = hyperparameters["beta_start"]
        self.epsilon = hyperparameters["epsilon"]
        # double ended queue
        self.buffer = [None] * self.capacity
        self.probs = [0.0] * self.capacity
        self.data_pointer = 0
        self.size = 0
        self.max_prob = 1.0

    # TODO: Implement the add method
    def add(self, transition):
        # Add a transition to the buffer
        # transition = (state, action, reward, next_state, done)
        self.buffer[self.data_pointer] = transition
        self.probs[self.data_pointer] = self.max_prob 
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # TODO: Implement the sample method
    def sample(self, size):

        p_alpha = np.array(self.probs[:self.size]) ** self.alpha
        p_sum = p_alpha.sum()
        probs = p_alpha / p_sum
        indices = np.random.choice(self.size, size, p=probs)
        # Sample transitions based  on the sampled indices

        states = np.array([self.buffer[idx][0] for idx in indices], dtype=np.float32)
        actions = np.array([self.buffer[idx][1] for idx in indices], dtype=np.int64)
        rewards = np.array([self.buffer[idx][2] for idx in indices], dtype=np.float32)
        next_states = np.array([self.buffer[idx][3] for idx in indices], dtype=np.float32)
        dones = np.array([self.buffer[idx][4] for idx in indices], dtype=np.float32)

        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return (states, actions, rewards, next_states, dones), indices, weights


    def update_probs(self, indices, td_errors):
        # Update the priorities of the sampled transitions
        for idx, td_error in zip(indices, td_errors):
            new_prob = abs(td_error) + self.epsilon
            self.probs[idx] = new_prob
            self.max_prob = max(self.max_prob, new_prob)

class DQNAgent:
    def __init__(self, state_size, action_size, hyperparameters):
        # TODO: Initialize some parameters, networks, optimizer, replay buffer, etc.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: ", self.device)
        self.replay_buffer = ReplayBuffer(hyperparameters["replay_buffer_hyperparameters"])
        self.gamma = hyperparameters["gamma"]
        self.learning_rate = hyperparameters["learning_rate"]
        self.batch_size = hyperparameters["batch_size"]
        self.update_target_frequency = hyperparameters["update_target_every"]
        self.state_size = state_size
        self.action_size = action_size

        self.update_target_counter = 0
        self.model = DuelNet(state_size, action_size).to(self.device)
        self.target_model = DuelNet(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


    def get_action(self, state):
        # TODO: Implement the action selection
      
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # Get Q-values from the model
        q_values = self.model(state)
        # Select the action with the highest Q-value
        return torch.argmax(q_values).item()

    def update(self, target=None, learning=None):
        # TODO: Implement hard update or soft update
        # Hard update
        self.target_model.load_state_dict(self.model.state_dict())
        return

    def train(self):
        # TODO: Sample a batch from the replay buffer
        self.update_target_counter += 1

        (states, actions, rewards, next_states, dones), indices, weights = self.replay_buffer.sample(self.batch_size)
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)
        weights = torch.from_numpy(weights).to(self.device)
        # TODO: Compute loss and update the model

        # target network do not need to be updated
        # cumulative optimistic bias
        with torch.no_grad():
            next_actions = torch.argmax(self.model(next_states), dim=1)
            y_true = rewards + self.gamma * (1 - dones) * self.target_model(next_states).gather(dim=1, index=next_actions.unsqueeze(1)).squeeze()

        # for action, it need to align the dimension with other, so we need to unsqueeze
        y_pred = self.model(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze()
        loss = (0.5 * (y_pred - y_true)**2 * weights).mean()
        td_errors = (y_pred - y_true).detach().cpu().numpy()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # TODO: Update target network periodically
        self.replay_buffer.update_probs(indices, td_errors)
        if self.update_target_counter % self.update_target_frequency == 0:
            self.update()
            print("Target network updated")


