import gym
import os
import random
from itertools import chain

import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
import cv2

from model import *

from ppo_agent import MlpActorAgent
from utils import make_train_data
import torch.optim as optim
from torch.multiprocessing import Pipe, Process

from collections import deque
from sklearn.utils import shuffle
from tensorboardX import SummaryWriter
from torch.distributions.categorical import Categorical

class Environment(Process):
    def __init__(
            self,
            is_render,
            env_idx,
            child_conn):
        super(Environment, self).__init__()
        self.daemon = True
        self.env = gym.make('LunarLander-v2')
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.history = np.zeros([4, 84, 84])

        self.reset()

    def run(self):
        super(Environment, self).run()
        while True:
            action = self.child_conn.recv()
            if self.is_render:
                self.env.render()

            obs, reward, done, info = self.env.step(action)

            self.rall += reward
            self.steps += 1

            if done:
                self.history = self.reset()

            self.child_conn.send(
                [obs, reward, done, info])

    def reset(self):
        self.steps = 0
        self.episode += 1
        self.rall = 0
        obs = self.env.reset()
        return obs

if __name__ == '__main__':

    writer = SummaryWriter()
    use_cuda = True
    use_gae = True
    is_load_model = False
    is_render = False
    use_standardization = True
    lr_schedule = False
    life_done = True
    use_noisy_net = True

    num_worker = 4

    num_step = 128
    ppo_eps = 0.1
    epoch = 3
    batch_size = 32
    max_step = 1.15e8

    learning_rate = 0.00025

    stable_eps = 1e-30
    epslion = 0.1
    entropy_coef = 0.01
    alpha = 0.99
    gamma = 0.99
    clip_grad_norm = 0.5

    agent = MlpActorAgent(
        8,
        4,
        num_step,
        gamma,
        use_cuda=use_cuda,
        use_gae=use_gae)

    if is_load_model:
        agent.model.load_state_dict(torch.load(model_path))

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = Environment(is_render, idx, child_conn)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, 8])

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    global_step = 0
    recent_prob = deque(maxlen=10)
    score = 0
    while True:
        num_rollout += 1
        total_state, total_reward, total_done, total_next_state, total_action = [], [], [], [], []
        global_step += (num_worker * num_step)

        for _ in range(num_step):
            actions = agent.get_action(states)

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            next_states, rewards, dones, real_dones = [], [], [], []
            for parent_conn in parent_conns:
                s, r, d, _ = parent_conn.recv()
                next_states.append(s)
                rewards.append(r)
                dones.append(d)

            score += rewards[sample_env_idx]
            next_states = np.vstack(next_states)
            rewards = np.hstack(rewards)
            dones = np.hstack(dones)

            total_state.append(states)
            total_next_state.append(next_states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)

            states = next_states[:, :]

            if dones[sample_env_idx]:
                sample_episode += 1
                if sample_episode < 333:
                    print('episodes:', sample_episode, '| score:', score)
                    writer.add_scalar('data/reward', score, sample_episode)
                    score = 0
        total_state = np.stack(total_state).transpose(
            [1, 0, 2]).reshape([-1, 8])
        total_next_state = np.stack(total_next_state).transpose(
            [1, 0, 2]).reshape([-1, 8])
        total_reward = np.stack(total_reward).transpose().reshape([-1])
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_done = np.stack(total_done).transpose().reshape([-1])
        value, next_value, policy = agent.forward_transition(
            total_state, total_next_state)
        total_target = []
        total_adv = []
        for idx in range(num_worker):
            target, adv = make_train_data(total_reward[idx * num_step:(idx + 1) * num_step],
                                          total_done[idx * num_step:(idx + 1) * num_step],
                                          value[idx * num_step:(idx + 1) * num_step],
                                          next_value[idx * num_step:(idx + 1) * num_step])
            # print(target.shape)
            total_target.append(target)
            total_adv.append(adv)

        print('training')
        agent.train_model(
            total_state,
            np.hstack(total_target),
            total_action,
            np.hstack(total_adv))