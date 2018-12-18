from envs import *
from utils import *
from config import *
from ppo_agent import *
from torch.multiprocessing import Pipe
from tensorboardX import SummaryWriter
from mlagents.envs import UnityEnvironment
import numpy as np
import time

env = UnityEnvironment(file_name='pushblock/pushblock')
default_brain = env.brain_names[0]
brain = env.brains[default_brain]
env.reset()
num_worker = 32
input_size = 210
output_size = 5
num_step = 256
gamma = 0.99
pre_obs_norm_step = 10000

reward_rms = RunningMeanStd()
obs_rms = RunningMeanStd(1, input_size)
discounted_reward = RewardForwardFilter(gamma)
agent = MlpICMAgent(input_size, output_size, num_worker,
                    num_step, gamma, use_cuda=True)


steps = 0
next_obs = []
print('Start to initialize observation normalization ...')
while steps < pre_obs_norm_step:
    steps += num_worker
    actions = np.random.randint(output_size, size=num_worker)
    env_info = env.step(actions)[default_brain]
    obs = env_info.vector_observations
    for o in obs:
        next_obs.append(o)
    print('initializing...:', steps, '/', pre_obs_norm_step)
next_obs = np.stack(next_obs)
obs_rms.update(next_obs)
print('End to initialize')

writer = SummaryWriter()
writer_iter = 2278
global_update = 0
global_step = 0
sample_i_rall = 0
sample_episode = 0
sample_env_idx = 0
sample_rall = 0
states = np.zeros([num_worker, input_size])
while True:
    total_state, total_reward, total_done, total_next_state, \
    total_action, total_int_reward, total_next_obs, total_values,\
    total_policy = [], [], [], [], [], [], [], [], []
    global_step += (num_step * num_worker)
    global_update += 1

    for _ in range(num_step):
        actions, value, policy = agent.get_action((np.float32(states) - obs_rms.mean)/np.sqrt(obs_rms.var))

        env_info = env.step(actions)[default_brain]

        next_states, rewards, dones, real_dones, next_obs = [], [], [], [], []

        obs = env_info.vector_observations
        reward = env_info.rewards
        reward = np.clip(reward, 0, 1)
        done = env_info.local_done

        for o, r, d in zip(obs, reward, done):
            next_states.append(o)
            rewards.append(r)
            dones.append(d)
        
        next_states = np.stack(next_states)
        rewards = np.hstack(rewards)
        dones = np.hstack(dones)

        intrinsic_reward = agent.compute_intrinsic_reward(
                            (states - obs_rms.mean)/np.sqrt(obs_rms.var),
                            (next_states - obs_rms.mean)/np.sqrt(obs_rms.var),
                            actions)

        intrinsic_reward = np.hstack(intrinsic_reward)

        sample_i_rall += intrinsic_reward[sample_env_idx]
        sample_rall += rewards[sample_env_idx]

        total_int_reward.append(intrinsic_reward)
        total_state.append(states)
        total_next_state.append(next_states)
        total_reward.append(rewards)
        total_done.append(dones)
        total_action.append(actions)
        total_values.append(value)
        total_policy.append(policy)

        states = next_states[:, :]

        if dones[sample_env_idx]:
            sample_episode += 1
            if sample_episode < writer_iter:
                writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
                writer.add_scalar('data/int_reward_per_epi', sample_i_rall, sample_episode)
            print("[Episode {}] rall: {}  i_: {}".format(
                    sample_episode, sample_rall, sample_i_rall))
            sample_i_rall = 0
            sample_rall = 0

    _, value, _ = agent.get_action((np.float32(states) - obs_rms.mean) / np.sqrt(obs_rms.var))
    total_values.append(value)

    total_state = np.stack(total_state).transpose([1, 0, 2]).reshape([-1, input_size])
    total_next_state = np.stack(total_next_state).transpose([1, 0, 2]).reshape([-1, input_size])
    total_action = np.stack(total_action).transpose().reshape([-1])
    total_reward = np.stack(total_reward).transpose()
    total_done = np.stack(total_done).transpose()
    total_values = np.stack(total_values).transpose()
    total_logging_policy = np.vstack(total_policy)

    total_int_reward = np.stack(total_int_reward).transpose()
    total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                        total_int_reward.T])
    mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
    reward_rms.update_from_moments(mean, std ** 2, count)

    total_int_reward /= np.sqrt(reward_rms.var)
    
    if sample_episode < writer_iter:
        writer.add_scalar('data/int_reward_per_rollout', np.sum(total_int_reward) / num_worker, global_update)
        writer.add_scalar('data/max_prob', softmax(total_logging_policy).max(1).mean(), sample_episode)

    target, adv = make_train_data_icm(total_int_reward,
                                    np.zeros_like(total_int_reward),
                                    total_values,
                                    gamma,
                                    num_step,
                                    num_worker)

    adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)

    ext_target, ext_adv = make_train_data_icm(total_reward,
                                            total_done, total_values, gamma, num_step, num_worker)

    int_ratio = 0
    adv = adv * int_ratio + ext_adv * (1-int_ratio)
    target = target * int_ratio + ext_target * (1-int_ratio)

    #obs_rms.update(total_next_state)
    #print(obs_rms.count)
    print('training')
    agent.train_model((np.float32(total_state) - obs_rms.mean) / np.sqrt(obs_rms.var),
                        (np.float32(total_next_state) - obs_rms.mean) / np.sqrt(obs_rms.var),
                        target, total_action,
                        adv, total_policy)