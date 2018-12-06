from mlagents.envs import UnityEnvironment
import numpy as np
import time

env = UnityEnvironment(file_name='pyramid_16')
default_brain = env.brain_names[0]
brain = env.brains[default_brain]

env_info = env.reset(train_mode=True)[default_brain]

for episode in range(10):
    done = False
    step = 0
    while not done:
        step += 1
        action = np.random.randint(5, size=16)
        env_info = env.step(action)[default_brain]
        done = env_info.local_done[0]
        print(done, step)
        print(np.array(env_info.rewards).shape)
        print(np.array(env_info.vector_observations).shape)
        print(env_info.local_done)
        print('---')
        time.sleep(0.01)