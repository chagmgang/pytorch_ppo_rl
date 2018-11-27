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

class ICMModule(nn.Module):
    def __init__(self):
        super(ICMModule, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(7*7*128, 256)
        )

        self.inverse_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

        self.forward_net_1 = nn.Sequential(
            nn.Linear(3 + 256, 256),
            nn.LeakyReLU()
        )

        self.forward_net_2 = nn.Sequential(
            nn.Linear(3 + 256, 256),
        )

        self.residual_1 = nn.Sequential(
            nn.Linear(3 + 256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256)
        )
        
        self.residual_2 = nn.Sequential(
            nn.Linear(3 + 256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256)
        )

    def forward(self, input):
        state, next_state, action = input
        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)

        concat_encode = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(concat_encode)
    
        concat_action_encode_state = torch.cat((action, encode_state), 1)
        pred_next_state = self.forward_net_1(concat_action_encode_state)

        pred_next_state_orig = torch.cat((action, pred_next_state), 1)
        pred_next_state = self.residual_1(pred_next_state_orig)
        pred_next_state_orig = torch.cat((action, pred_next_state), 1) + pred_next_state_orig
        pred_next_state = self.residual_2(pred_next_state_orig)
        pred_next_state_orig = torch.cat((action, pred_next_state), 1) + pred_next_state_orig
        pred_next_state = self.forward_net_2(pred_next_state_orig)

        return encode_next_state, pred_next_state, pred_action