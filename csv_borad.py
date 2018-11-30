from tensorboardX import SummaryWriter
import numpy as np

ext_0 = np.genfromtxt('ext_0.csv')[:193]
ext_1 = np.genfromtxt('ext_1.csv')[:193]
ext_2 = np.genfromtxt('ext_2.csv')[:193]
ext_int_0 = np.genfromtxt('ext_int_0.csv')[:193]
ext_int_1 = np.genfromtxt('ext_int_1.csv')[:193]
ext_int_2 = np.genfromtxt('ext_int_2.csv')[:193]

ext = np.array([ext_0, ext_1, ext_2]).transpose()
ext_int = np.array([ext_int_0, ext_int_1, ext_int_1]).transpose()
average_ext = np.mean(ext, axis=1)
average_ext_int = np.mean(ext_int, axis=1)

writer = SummaryWriter()
episode = 0
for e, e_i in zip(average_ext, average_ext_int):
    episode += 1
    #writer.add_scalar('data/reward', e, episode)
    writer.add_scalar('data/reward', e_i, episode)