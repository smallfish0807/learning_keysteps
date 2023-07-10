import random

import torch
import torch.nn as nn

from layer import Lambda


class DeepQN(nn.Module):
    def __init__(self,
                 device,
                 input_shape,
                 num_actions,
                 exploration,
                 use_cnn=False,
                 dropout=0.):
        super().__init__()

        self.device = device
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.exploration = exploration

        if not use_cnn:
            self.features = nn.Sequential(nn.Linear(input_shape[0], 128),
                                          nn.ReLU(), nn.Linear(128, 128),
                                          nn.ReLU(), nn.Linear(128, 512),
                                          nn.ReLU())
        else:
            self.features = nn.Sequential(
                Lambda(lambda x: x / 255.0),
                nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(), nn.Conv2d(32, 64, kernel_size=4,
                                     stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
                Lambda(lambda x: x.view(x.size(0), -1)))

        if self.exploration['name'] == 'epsilon_greedy':
            if dropout > 1e-6:
                self.value = nn.Sequential(nn.Linear(self.feature_size(), 512),
                                           nn.ReLU(), nn.Dropout(dropout),
                                           nn.Linear(512, self.num_actions))
            else:
                self.value = nn.Sequential(nn.Linear(self.feature_size(), 512),
                                           nn.ReLU(),
                                           nn.Linear(512, self.num_actions))

        else:
            raise ValueError('Exploration strategy not implemented')

    def forward(self, x):
        x = self.features(x)
        x = self.value(x)
        return x

    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).numel()

    def values(self, state):
        state = torch.tensor(state, dtype=torch.float,
                             device=self.device).unsqueeze(0)
        output = self.forward(state)
        return output

    def act(self, state, epsilon=0.):
        if self.exploration['name'] == 'epsilon_greedy' and random.random(
        ) <= epsilon:
            action = random.randrange(self.num_actions)
        else:
            action = self.values(state).max(1)[1].cpu().numpy()[0]
        return action
