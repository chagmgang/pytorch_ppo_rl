import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init

class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)

def swish(x):
    return x * F.sigmoid(x)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class MlpActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(MlpActorCriticNetwork, self).__init__()
        self.com_layer1 = nn.Linear(input_size, 256)
        self.batch_1 = nn.BatchNorm1d(256)
        self.com_layer2 = nn.Linear(256, 256)
        self.batch_2 = nn.BatchNorm1d(256)
        self.com_layer3 = nn.Linear(256, 256)
        self.batch_3 = nn.BatchNorm1d(256)

        self.actor_1 = nn.Linear(256, 256)
        self.actor_2 = nn.Linear(256, 256)
        self.actor = nn.Linear(256, output_size)
        self.critic_1 = nn.Linear(256, 256)
        self.critic_2 = nn.Linear(256, 256)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        
        x = swish(self.batch_1(self.com_layer1(x)))
        x = swish(self.batch_2(self.com_layer2(x)))
        x = swish(self.batch_3(self.com_layer3(x)))
        actor_1 = swish(self.actor_1(x))
        actor_2 = swish(self.actor_2(x))
        policy = self.actor(actor_2)
        critic_1 = swish(self.critic_1(x))
        critic_2 = swish(self.critic_2(x))
        value = self.critic(critic_2)

        return policy, value



class MlpICMModel(nn.Module):
    def __init__(self, input_size, output_size, use_cuda=True):
        super(MlpICMModel, self).__init__()

        self.resnet_time = 4
        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.feature = nn.Sequential(
            nn.Linear(self.input_size, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, 256),
        )

        self.inverse_net = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.BatchNorm1d(256),
            Swish(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            Swish(),
            nn.Linear(256, self.output_size)
        )

        self.residual = [nn.Sequential(
            nn.Linear(self.output_size + 256, 256),
            Swish(),
            nn.Linear(256, 256),
        ).to(self.device)] * 2 * self.resnet_time

        self.forward_net_1 = nn.Sequential(
            nn.Linear(self.output_size + 256, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, 256)
        )
        self.forward_net_2 = nn.Sequential(
            nn.Linear(self.output_size + 256, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, 256)
        )

    def forward(self, inputs):
        state, next_state, action = inputs

        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)
        # get pred action
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(pred_action)
        # ---------------------

        # get pred next state
        pred_next_state_feature_orig = torch.cat((encode_state, action), 1)
        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig)

        # residual
        for i in range(self.resnet_time):
            pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](
                torch.cat((pred_next_state_feature, action), 1)) + pred_next_state_feature_orig

        pred_next_state_feature = self.forward_net_2(torch.cat((pred_next_state_feature_orig, action), 1))

        real_next_state_feature = encode_next_state
        return real_next_state_feature, pred_next_state_feature, pred_action

















class CnnActorCriticNetwork(nn.Module):
    def __init__(self):
        super(CnnActorCriticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, 512)

        self.actor = nn.Linear(512, 3)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value

class ICMCnnActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size, use_noisy_net=False):
        super(ICMCnnActorCriticNetwork, self).__init__()

        if use_noisy_net:
            linear = NoisyLinear
        else:
            linear = nn.Linear

        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            linear(
                7 * 7 * 64,
                512),
            nn.LeakyReLU()
        )

        self.actor = nn.Sequential(
            linear(512, 512),
            nn.LeakyReLU(),
            linear(512, output_size)
        )

        self.critic = nn.Sequential(
            linear(512, 512),
            nn.LeakyReLU(),
            linear(512, 1)
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for i in range(len(self.actor)):
            if type(self.actor[i]) == nn.Linear:
                init.orthogonal_(self.actor[i].weight, 0.01)
                self.actor[i].bias.data.zero_()

        for i in range(len(self.critic)):
            if type(self.critic[i]) == nn.Linear:
                init.orthogonal_(self.critic[i].weight, 0.01)
                self.critic[i].bias.data.zero_()

    def forward(self, state):
        x = self.feature(state)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value


class ICMModel(nn.Module):
    def __init__(self, input_size, output_size, use_cuda=True):
        super(ICMModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        feature_output = 7 * 7 * 64
        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512)
        )

        self.inverse_net = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

        self.residual = [nn.Sequential(
            nn.Linear(output_size + 512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
        ).to(self.device)] * 8

        self.forward_net_1 = nn.Sequential(
            nn.Linear(output_size + 512, 512),
            nn.LeakyReLU()
        )
        self.forward_net_2 = nn.Sequential(
            nn.Linear(output_size + 512, 512),
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, inputs):
        state, next_state, action = inputs

        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)
        # get pred action
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(pred_action)
        # ---------------------

        # get pred next state
        pred_next_state_feature_orig = torch.cat((encode_state, action), 1)
        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig)

        # residual
        for i in range(4):
            pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](
                torch.cat((pred_next_state_feature, action), 1)) + pred_next_state_feature_orig

        pred_next_state_feature = self.forward_net_2(torch.cat((pred_next_state_feature_orig, action), 1))

        real_next_state_feature = encode_next_state
        return real_next_state_feature, pred_next_state_feature, pred_action