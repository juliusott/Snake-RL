import torch
import torch.nn as nn
import torch.optim as optim
from Globals import *
from collections import deque
from collections import namedtuple
import matplotlib.pyplot as plt

F = nn.functional

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class Actor_Critic(nn.Module):
    def __init__(self):
        super(Actor_Critic, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, (8, 8), stride=2)
        self.batch1 = nn.BatchNorm2d(32)
        self.max = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(32, 64, (5, 5), stride=2)
        self.batch2 = nn.BatchNorm2d(64)
        self.out_actor = nn.Linear(64,4)
        self.out_critic = nn.Linear(64,1)

    def forward(self, x):
        x = self.max(self.batch1(F.leaky_relu(self.conv1(x))))
        x = self.max(self.batch2(F.leaky_relu(self.conv2(x))))
        x = x.view(x.size(0), -1)
        value = self.out_critic(x)
        policy = self.out_actor(x)

        return policy, value



def possible_actions(state):
    _possible_actions = [True, True, True, True]
    for i, direct in enumerate([Point(0, -1), Point(0, 1), Point(-1, 0), Point(1, 0)]):
        if state.snake_length > 1 and state.snake[0] + direct == state.snake[1]:
            _possible_actions[i] = False
    return _possible_actions
