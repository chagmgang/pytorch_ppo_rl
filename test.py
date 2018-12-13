from mlagents.envs import UnityEnvironment
import numpy as np
import time

env = UnityEnvironment(file_name='pushblock/pushblock')
default_brain = env.brain_names[0]
brain = env.brains[default_brain]

env_info = env.reset(train_mode=True)[default_brain]
print(env_info.vector_observations.shape)

# 0 - no action, 1 - forward, 2 - backward, 3 - turn right, 4 - turn left


for episode in range(10):
    done = False
    step = 0
    while not done:
        step += 1
        action = [4 for i in range(32)]
        #action = np.random.randint(5, size=32)
        env_info = env.step(action)[default_brain]
        done = env_info.local_done[0]
        time.sleep(0.01)