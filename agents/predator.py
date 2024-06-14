# File: multiagent-predator-prey/agents/predator.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Predator(nn.Module):
    def __init__(self, input_dim, action_dim, algorithm='COMIX'):
        super(Predator, self).__init__()
        self.algorithm = algorithm
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

        if self.algorithm in ['MADDPG', 'FacMADDPG']:
            self.actor_network = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
                nn.Tanh()
            )
            self.target_actor_network = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
                nn.Tanh()
            )
            self.critic_network = nn.Sequential(
                nn.Linear(input_dim + action_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            self.target_critic_network = nn.Sequential(
                nn.Linear(input_dim + action_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=0.001)
            self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=0.001)

    def forward(self, x):
        return self.fc(x)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(state).float()  # Ensure state is a torch tensor
            if self.algorithm == 'COMIX':
                action = self.forward(state)
            elif self.algorithm in ['MADDPG', 'FacMADDPG']:
                action = self.actor_network(state)
        return action.numpy()  # Return a numpy array for the environment

    def update_q_network(self, batch):
        if self.algorithm == 'COMIX':
            states, actions, rewards, next_states, dones = batch
            q_values = self.q_network(torch.cat([states, actions], dim=1))
            next_actions = self.forward(next_states)
            next_q_values = self.target_q_network(torch.cat([next_states, next_actions], dim=1)).detach()
            target_q_values = rewards + (1 - dones) * 0.99 * next_q_values  # gamma = 0.99

            loss = self.loss_fn(q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_target_network(self, tau=0.01):
        if self.algorithm == 'COMIX':
            self.target_q_network.load_state_dict(self.q_network.state_dict())
        elif self.algorithm in ['MADDPG', 'FacMADDPG']:
            for target_param, param in zip(self.target_actor_network.parameters(), self.actor_network.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            for target_param, param in zip(self.target_critic_network.parameters(), self.critic_network.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def update_critic_network(self, batch):
        if self.algorithm in ['MADDPG', 'FacMADDPG']:
            states, actions, rewards, next_states, dones = batch
            next_actions = self.target_actor_network(next_states)
            next_q_values = self.target_critic_network(torch.cat([next_states, next_actions], dim=1)).detach()
            target_q_values = rewards + (1 - dones) * 0.99 * next_q_values  # gamma = 0.99

            q_values = self.critic_network(torch.cat([states, actions], dim=1))
            critic_loss = self.loss_fn(q_values, target_q_values)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

    def update_actor_network(self, batch, all_agents):
        if self.algorithm in ['MADDPG', 'FacMADDPG']:
            states, actions, _, _, _ = batch
            policy_loss = -self.critic_network(torch.cat([states, self.actor_network(states)], dim=1)).mean()
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()
