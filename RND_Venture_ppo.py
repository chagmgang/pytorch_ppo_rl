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
import matplotlib.pyplot as plt

import torch.optim as optim
from torch.multiprocessing import Pipe, Process

from collections import deque
from sklearn.utils import shuffle
from ppo_agent import CNNRNDAgent, CNNActorAgent, make_train_data_rnd, RunningMeanStd, RewardForwardFilter
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
        self.env = gym.make('VentureNoFrameskip-v4')
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.history = np.zeros([4, 84, 84])
        self.obs_buffer = np.zeros((2,84,84))
        self.reset()
        self.frame_skip = 4
        self.lives = self.env.env.ale.lives()
        self.max_step_per_episode = 18000

    def run(self):
        super(Environment, self).run()
        while True:
            action = self.child_conn.recv()
            total_reward = 0
            for i in range(self.frame_skip):
                obs, reward, done, info = self.env.step(action)

                self.obs_buffer[0, :] = self.obs_buffer[1, :]
                self.obs_buffer[1, :] = self.pre_proc(
                    self.env.env.ale.getScreenGrayscale().squeeze().astype('float32'))
                max_frame = self.obs_buffer.max(axis=0)
                '''
                print(max_frame.shape)
                plt.subplot(311)
                plt.imshow(self.obs_buffer[0], cmap='gray')
                plt.subplot(312)
                plt.imshow(self.obs_buffer[1], cmap='gray')
                plt.subplot(313)
                plt.imshow(max_frame, cmap='gray')
                plt.show()
                '''
                total_reward += reward
                if done:
                    break

            if self.max_step_per_episode < self.steps:
                done = True
            
            force_done = done

            if self.is_render:
                self.env.render()
            
            self.history[:3, :, :] = self.history[1:, :, :]
            self.history[3, :, :] = max_frame

            '''
            if self.steps > 20:
                plt.subplot(211)
                plt.imshow(self.history[2], cmap='gray')
                plt.subplot(212)
                plt.imshow(self.history[3], cmap='gray')
                plt.show()
            '''

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
        self.get_init_state(
            self.env.env.ale.getScreenGrayscale().squeeze().astype('float32'))
        return self.history[:, :, :]

    def pre_proc(self, X):
        x = cv2.resize(X, (84, 84))
        x *= (1.0 / 255.0)

        return x

    def get_init_state(self, s):
        for i in range(4):
            self.history[i, :, :] = self.pre_proc(s)

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

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
    discounted_reward = RewardForwardFilter(gamma)
    agent = CNNRNDAgent(num_step)
    #agent.predic_model = torch.load('predict_model.pt')
    #agent.target_model = torch.load('target_model.pt')
    #agent.model = torch.load('model.pt')

    #if is_load_model:
    #    agent.model.load_state_dict(torch.load(model_path))

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

    states = np.zeros([num_worker, 4, 84, 84])

    num_rollout = 0
    training_start = 50
    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    global_step = 0
    recent_prob = deque(maxlen=10)
    score = 0
    int_score = 0

    while True:
        x, y = [], []
        num_rollout += 1
        total_state, total_reward, total_done,\
        total_next_state, total_action, total_int_reward,\
        total_next_obs, total_ext_values, total_int_values, total_policy = [], [], [], [], [], [], [], [], [], []
        global_step += (num_worker * num_step)
        env_step = 0
        for _ in range(num_step):
            env_step += 1
            actions, value_ext, value_int, policy = agent.get_action(states)

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            next_states, rewards, dones, real_dones, next_obs = [], [], [], [], []
            for parent_conn in parent_conns:
                s, r, d, rd = parent_conn.recv()

                next_states.append(s)
                rewards.append(r)
                dones.append(d)
                real_dones.append(rd)
                next_obs.append(s[3, :, :].reshape(1, 84, 84))
            
            next_states = np.stack(next_states)
            rewards = np.hstack(rewards)
            dones = np.hstack(dones)
            real_dones = np.hstack(real_dones)
            next_obs = np.stack(next_obs)

            normalized_next_obs = ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)
            intrinsic_reward = agent.get_intrinsic_reward(normalized_next_obs)
            intrinsic_reward = np.hstack(intrinsic_reward)

            '''
            x.append(env_step)
            y.append(intrinsic_reward[sample_env_idx])
            plt.step(x, y)
            plt.draw()
            plt.pause(0.001)
            plt.clf()
            '''

            total_next_obs.append(next_obs)
            total_int_reward.append(intrinsic_reward)
            total_state.append(states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)
            total_ext_values.append(value_ext)
            total_int_values.append(value_int)
            total_policy.append(policy)

            int_score += intrinsic_reward[sample_env_idx]
            score += rewards[sample_env_idx]
            if real_dones[sample_env_idx]:
                sample_episode += 1
                if sample_episode < 99999999999999999999:
                    print('episodes:', sample_episode, '| score:', score, '| int_score:', int_score)
                    writer.add_scalar('data/reward', score, sample_episode)
                    writer.add_scalar('data/int_reward', int_score, sample_episode)
                    score = 0
                    int_score = 0

            states = next_states[:, :, :, :]
        
        _, value_ext, value_int, _ = agent.get_action(states)
        total_ext_values.append(value_ext)
        total_int_values.append(value_int)

        total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
        total_reward = np.stack(total_reward).transpose().clip(-1, 1)
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_done = np.stack(total_done).transpose()
        total_next_obs = np.stack(total_next_obs).transpose([1, 0, 2, 3, 4]).reshape([-1, 1, 84, 84])
        total_ext_values = np.stack(total_ext_values).transpose()
        total_int_values = np.stack(total_int_values).transpose()
        total_logging_policy = np.vstack(total_policy)

        total_int_reward = np.stack(total_int_reward).transpose()
        total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                         total_int_reward.T])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)
        total_int_reward /= np.sqrt(reward_rms.var)

        ext_target, ext_adv = make_train_data_rnd(total_reward,
                                                    total_done,
                                                    total_ext_values,
                                                    gamma,
                                                    num_step,
                                                    num_worker)

        int_target, int_adv = make_train_data_rnd(total_int_reward,
                                              np.zeros_like(total_int_reward),
                                              total_int_values,
                                              gamma,
                                              num_step,
                                              num_worker)
        int_coef = 1
        ext_coef = 1
        total_adv = int_adv * int_coef + ext_adv * ext_coef
        obs_rms.update(total_next_obs)

        print('rollout: ', num_rollout)
        if num_rollout > training_start:
            print('training')
            agent.train_model(total_state, ext_target, int_target, total_action,
                            total_adv, ((total_next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5),
                            total_policy)
            torch.save(agent.predic_model, 'predict_model.pt')
            torch.save(agent.target_model, 'target_model.pt')
            torch.save(agent.model, 'model.pt')
        