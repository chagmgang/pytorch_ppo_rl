# This Repository is Reinforcement Learning related with PPO

This Repository is Reinforcece Learning Implementation related with PPO.
The framework used in this Repository is Pytorch. The multi-processing method is basically built in. The agents are trained by PAAC(Parallel Advantage Actor Critic) strategy.  

## 1. Multi-processing MLP Proximal Policy Optimization  

* Script : LunarLander_ppo.py  
* Environment : LunarLander-v2  
* Orange : 8 Process, Blue : 4 Process, Red : 1 Process

###### LunarLander-v2
<div align="center">
  <img src="source/lunarlander.gif" width="50%" height='300'><img src="source/lunarlander_result.png" width="50%" height='300'>
</div>


## 2. Multi-processing CNN Proximal Policy Opimization


* Script : Breakout_ppo.py
* Environment : BreakoutDeterministic-v4
* Red: 8 Process, Blue: 4 Process, Orange: 1 Process

###### BreakoutDeterministic-v4
<div align="center">
  <img src="source/breakout_0.gif" width="24%" height='300'>
  <img src="source/breakout_1.gif" width="25%" height='300'>
  <img src="source/breakout_result.png" width="50%" height='300'>
</div>

# 3. Multi-processing CNN Proximal Policy Opitimization with Intrinsic Curiosity Module

* Script : Breakout_ppo_icm.py
* Environment : BreakoutNoFrameskip-v4(handled by custom environment)
* With no environment Reward
* Because the game initial key is not selected, the peak point and performance drop is generated.

###### BreakoutNoFrameskip-v4(handled by custom environment)
<div align="center">
  <img src="source/breakout_result_icm.png" width="50%" height='300'>
</div>

## Reference
[1] [mario_rl](https://github.com/jcwleo/mario_rl)

[2] [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)

[2] [Efficient Parallel Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1705.04862)

[3] [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

[4] [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363)

[5] [Large-Scale Study of Curiosity-Driven Learning](https://arxiv.org/abs/1808.04355)

[6] [curiosity-driven-exploration-pytorch](https://github.com/jcwleo/curiosity-driven-exploration-pytorch)