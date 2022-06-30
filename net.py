import torch
import torch.nn as nn
import torch.nn.functional as F

import random

class Qnet(nn.Module):
    def __init__(self, hidden_dim):
        super(Qnet, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, state):
        return self.nn(state)

    def sample_action(self, state, eps):
        action = self.forward(state)
        if random.random() < eps:
            return random.choice([i for i in range(4)])
        else:
            return action.argmax().item()
