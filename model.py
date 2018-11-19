import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class CnnRND(nn.Module):
    def __init__(self):
        super(CnnRND, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1)
        self.fc1 = nn.Linear(7*7*128, 7*128)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 7*7*128)
        x = F.relu(self.fc1(x))
        return x

class MlpActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(MlpActorCriticNetwork, self).__init__()
        self.com_layer1 = nn.Linear(input_size, 100)
        self.com_layer2 = nn.Linear(100, 100)

        self.actor = nn.Linear(100, output_size)
        self.critic = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.com_layer1(x))
        x = F.relu(self.com_layer2(x))
        policy = self.actor(x)
        value = self.critic(x)

        return policy, value

class CnnActorCriticNetwork(nn.Module):
    def __init__(self):
        super(CnnActorCriticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1)
        self.fc1 = nn.Linear(7*7*128, 7*128)
        self.fc2 = nn.Linear(7*128, 256)

        self.actor = nn.Linear(256, 3)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 7*7*128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value