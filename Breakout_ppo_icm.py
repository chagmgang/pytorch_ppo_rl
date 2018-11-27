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

import torch.optim as optim
from torch.multiprocessing import Pipe, Process

from collections import deque
from sklearn.utils import shuffle
from ppo_agent import ICMCNNActorAgent, make_train_data_icm, RunningMeanStd, RewardForwardFilter
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
        self.env = gym.make('BreakoutNoFrameskip-v4')
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.history = np.zeros([4, 84, 84])

        self.reset()
        self.lives = self.env.env.ale.lives()

    def run(self):
        super(Environment, self).run()
        while True:
            action = self.child_conn.recv()
            if self.is_render:
                self.env.render()

            _, reward, done, info = self.env.step(action + 1)

            if life_done:
                if self.lives > info['ale.lives'] and info['ale.lives'] > 0:
                    force_done = True
                    self.lives = info['ale.lives']
                else:
                    force_done = done
            else:
                force_done = done

            if force_done:
                reward = -1
                
            self.history[:3, :, :] = self.history[1:, :, :]
            self.history[3, :, :] = self.pre_proc(
                self.env.env.ale.getScreenGrayscale().squeeze().astype('float32'))

            self.rall += reward
            self.steps += 1

            if done:
                self.history = self.reset()

            self.child_conn.send(
                [self.history[:, :, :], reward, force_done, done])

    def reset(self):
        self.steps = 0
        self.episode += 1
        self.rall = 0
        self.env.reset()
        self.lives = self.env.env.ale.lives()
        self.env.step(1)
        self.get_init_state(
            self.env.env.ale.getScreenGrayscale().squeeze().astype('float32'))
        return self.history[:, :, :]

    def pre_proc(self, X):
        x = cv2.resize(X, (84, 84))

        return x

    def get_init_state(self, s):
        for i in range(4):
            self.history[i, :, :] = self.pre_proc(s)

if __name__ == '__main__':

    writer = SummaryWriter()
    use_cuda = True
    use_gae = True
    is_load_model = False
    is_render = True
    use_standardization = True
    lr_schedule = False
    life_done = True
    use_noisy_net = True

    num_worker = 16

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

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1,1,84,84))
    discounted_reward = RewardForwardFilter(0.99)
    agent = ICMCNNActorAgent(
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

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    int_score = 0
    global_step = 0
    score = 0
    pre_obs_norm_step = 10000

    
    print('Start to initailize observation normalization parameter.....')
    next_obs = []
    steps = 0
    while steps < pre_obs_norm_step:
        steps += num_worker
        actions = np.random.randint(0, 3, size=(num_worker,))

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        for parent_conn in parent_conns:
            s, r, d, rd = parent_conn.recv()
            next_obs.append(s[3, :, :].reshape([1, 84, 84]))
        print('initializing ...', steps, '/', pre_obs_norm_step)
    next_obs = np.stack(next_obs)
    obs_rms.update(next_obs)
    print('End to initalize...')
    

    states = np.zeros([num_worker, 4, 84, 84])
    
    while True:
        total_state, total_reward, total_done, total_values, \
            total_next_state, total_action, total_int_reward, total_policy = [], [], [], [], [], [], [], []
        global_step += (num_worker * num_step)

        for _ in range(num_step):
            actions, value, policy = agent.get_action((np.float32(states)-obs_rms.mean)/np.sqrt(obs_rms.var))

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            next_states, rewards, dones, real_dones = [], [], [], []
            for parent_conn in parent_conns:
                s, r, d, rd = parent_conn.recv()
                next_states.append(s)
                rewards.append(r)
                dones.append(d)
                real_dones.append(rd)

            score += rewards[sample_env_idx]
            next_states = np.stack(next_states)
            rewards = np.hstack(rewards)
            dones = np.hstack(dones)
            real_dones = np.hstack(real_dones)

            intrinsic_reward = agent.intrinsic_reward((states-obs_rms.mean)/np.sqrt(obs_rms.var),
                    (next_states-obs_rms.mean)/np.sqrt(obs_rms.var), actions)
            intrinsic_reward = np.hstack(intrinsic_reward)
            print(intrinsic_reward)

            int_score += intrinsic_reward[sample_env_idx]
            total_int_reward.append(intrinsic_reward)
            total_state.append(states)
            total_next_state.append(next_states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)
            total_policy.append(policy)
            total_values.append(value)

            states = next_states[:, :, :, :]

            if real_dones[sample_env_idx]:
                sample_episode += 1
                writer.add_scalar('data/reward', score, sample_episode)
                writer.add_scalar('data/int_reward', int_score, sample_episode)
                print('episode :', sample_episode, '| score :', score, '| int_score :', int_score)
                score = 0
                int_score = 0

        _, value, _ = agent.get_action(np.float32(states) - obs_rms.mean / np.sqrt(obs_rms.var))
        total_values.append(value)

        total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
        total_next_state = np.stack(total_next_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_done = np.stack(total_done).transpose()
        total_values = np.stack(total_values).transpose()
        total_logging_policy = np.vstack(total_policy)

        total_int_reward = np.stack(total_int_reward).transpose()
        total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                         total_int_reward.T])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)
        
        print('training')
        #print(total_int_reward)
        target, adv = make_train_data_icm(total_int_reward,
                                      np.zeros_like(total_int_reward),
                                      total_values,
                                      0.99,
                                      num_step,
                                      num_worker)

        #print(target)
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
        #print(adv)
        #print('---')
        agent.train_model(np.float32(total_state) - obs_rms.mean / np.sqrt(obs_rms.var),
                          np.float32(total_next_state) - obs_rms.mean / np.sqrt(obs_rms.var),
                          target, total_action, adv, total_policy)