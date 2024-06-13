# File: multiagent-predator-prey/agents/prey.py

import torch
import torch.nn as nn

class Prey(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Prey, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)  # Convert to float32
            action = self.forward(state)
        return action.numpy()
