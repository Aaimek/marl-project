# File: multiagent-predator-prey/agents/predator.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Predator(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Predator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Assuming action space is normalized to [-1, 1]
        )
        self.q_network = nn.Sequential(
            nn.Linear(input_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.target_q_network = nn.Sequential(
            nn.Linear(input_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.fc(x)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(state).float()  # Ensure state is a torch tensor
            action = self.forward(state)
        return action.numpy()  # Return a numpy array for the environment

    def update_q_network(self, batch):
        states, actions, rewards, next_states, dones = batch
        q_values = self.q_network(torch.cat([states, actions], dim=1))
        next_actions = self.forward(next_states)
        next_q_values = self.target_q_network(torch.cat([next_states, next_actions], dim=1)).detach()
        target_q_values = rewards + (1 - dones) * 0.99 * next_q_values  # gamma = 0.99

        # Debug prints to check ssvalues
        # print(f"States: {states}")
        # print(f"Actions: {actions}")
        # print(f"Rewards: {rewards}")
        # print(f"Next States: {next_states}")
        # print(f"Q-values: {q_values}")
        # print(f"Target Q-values: {target_q_values}")

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())
