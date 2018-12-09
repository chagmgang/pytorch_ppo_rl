from envs import *
from utils import *
from config import *
from ppo_agent import *
from torch.multiprocessing import Pipe
from tensorboardX import SummaryWriter
from mlagents.envs import UnityEnvironment
import numpy as np
import time

class Environment(Process):
    def __init__(self, is_render, env_idx, child_conn):
        super(Environment, self).__init__()
        self.daemon = True
        #self.env = gym.make('MountainCar-v0')
        self.env = UnityEnvironment(file_name='pyramid_linux_16/pyramid_linux')
        self.default_brain = self.env.brain_names[0]
        self.brain = self.env.brains[self.default_brain]
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.reset()
    
    def run(self):
        super(Environment, self).run()
        while True:
            action = self.child_conn.recv()
            '''
            a = []
            for i in action:
                if i == 0:
                    a.append(1)
                elif i == 1:
                    a.append(3)
                elif i == 2:
                    a.append(4)
            env_info = self.env.step(a)[self.default_brain]
            '''
            env_info = self.env.step(action)[self.default_brain]
            obs = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done

            self.steps += 1

            if done[0]:
                print("[Episode {}({})] Step: {}  Reward: {}".format(
                    self.episode, self.env_idx, self.steps, self.rall))
                obs = self.reset()
                self.rall = 0
                self.steps = 0
                self.episode += 1

            self.child_conn.send([obs, reward, done])

    def reset(self):
        self.steps = 0
        env_info = self.env.reset(train_mode=True)[self.default_brain]
        s = env_info.vector_observations
        return s
    
if __name__ == '__main__':
    real_num_worker = 1
    num_worker = 16
    works = []
    parent_conns = []
    child_conns = []
    pre_obs_norm_step = 10000
    
    input_size = 172
    output_size = 5
    num_step = 256
    gamma = 0.99
    is_render = True

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(1, 172)
    discounted_reward = RewardForwardFilter(gamma)
    agent = MlpICMAgent(input_size, output_size, num_worker,
                        num_step, gamma, use_cuda=True)

    print('make process')
    for idx in range(real_num_worker):
        parent_conn, child_conn = Pipe()
        work = Environment(is_render, idx, child_conn)
        work.start()
        works.append(work)
        child_conns.append(child_conn)
        parent_conns.append(parent_conn)
    print('make process done')
    
    
    steps = 0
    next_obs = []
    print('Start to initialize observation normalization ...')
    while steps < pre_obs_norm_step:
        steps += num_worker
        actions = np.random.randint(output_size, size=num_worker)
        actions = [actions]
        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        for parent_conn in parent_conns:
            s, r, d = parent_conn.recv()
            for i in range(num_worker):
                next_obs.append(s[i])
        print('initializing...:', steps, '/', pre_obs_norm_step)

    next_obs = np.stack(next_obs)
    obs_rms.update(next_obs)
    print('End to initialize')
    

    states = np.zeros([num_worker, 172])
    global_update = 0
    global_step = 0
    sample_i_rall = 0
    sample_episode = 0
    sample_env_idx = 0
    sample_rall = 0
    writer = SummaryWriter()

    while True:
        total_state, total_reward, total_done, total_next_state, \
        total_action, total_int_reward, total_next_obs, total_values,\
        total_policy = [], [], [], [], [], [], [], [], []
        global_step += (num_step * num_worker)
        global_update += 1

        for _ in range(num_step):
            actions, value, policy = agent.get_action((np.float32(states) - obs_rms.mean)/np.sqrt(obs_rms.var))
            a = [actions]
            for parent_conn, action in zip(parent_conns, a):
                parent_conn.send(action)
            next_states, rewards, dones, real_dones, next_obs = [], [], [], [], []
            for parent_conn in parent_conns:
                s, r, d = parent_conn.recv()
                for i in range(num_worker):
                    next_states.append(s[i])
                    rewards.append(r[i])
                    dones.append(d[i])
            
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
                writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
                sample_i_rall = 0
                sample_rall = 0

        _, value, _ = agent.get_action(np.float32(states) - obs_rms.mean / np.sqrt(obs_rms.var))
        total_values.append(value)

        total_state = np.stack(total_state).transpose([1, 0, 2]).reshape([-1, 172])
        total_next_state = np.stack(total_next_state).transpose([1, 0, 2]).reshape([-1, 172])
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_reward = np.stack(total_reward).transpose()
        total_done = np.stack(total_done).transpose()
        total_values = np.stack(total_values).transpose()
        total_logging_policy = np.vstack(total_policy)
        
        total_int_reward = np.stack(total_int_reward).transpose()
        total_int_reward *= 5
        total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                         total_int_reward.T])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)

        total_int_reward /= np.sqrt(reward_rms.var)
        
        writer.add_scalar('data/int_reward_per_epi', np.sum(total_int_reward)/num_worker, sample_episode)
        writer.add_scalar('data/int_reward_per_rollout', np.sum(total_int_reward) / num_worker, global_update)
        writer.add_scalar('data/max_prob', softmax(total_logging_policy).max(1).mean(), sample_episode)

        target, adv = make_train_data_icm(total_int_reward,
                                        np.zeros_like(total_int_reward),
                                        total_values,
                                        gamma,
                                        num_step,
                                        num_worker)

        adv = (adv - np.mean(adv) / np.std(adv) + 1e-8)

        obs_rms.update(total_next_state)
        #print(obs_rms.count)
        print('training')
        agent.train_model(np.float32(total_state) - obs_rms.mean / np.sqrt(obs_rms.var),
                          np.float32(total_next_state) - obs_rms.mean / np.sqrt(obs_rms.var),
                          target, total_action,
                          adv, total_policy)
