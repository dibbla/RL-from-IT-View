# trying to use DQN implementation from Tianshou
# this file is heavily based on the tianshou's basic demo
# another custom DQN would be required
import gymnasium as gym
import torch, numpy as np, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts

import minigrid
from minigrid.wrappers import ReseedWrapper
from minigrid.core.constants import OBJECT_TO_IDX
from custom_wrappers import SymbolicObsWrapperFullFlattenDirect

task = 'MiniGrid-SimpleCrossingS9N1-v0'
lr, epoch, batch_size = 1e-3, 10, 64
train_num, test_num = 10, 100
gamma, n_step, target_freq = 0.9, 3, 320
buffer_size = 20000
eps_train, eps_test = 0.1, 0.05
step_per_epoch, step_per_collect = 10000, 10
logger = ts.utils.TensorboardLogger(SummaryWriter('log/dqn'))

# you can also try with SubprocVectorEnv
train_envs = ts.env.DummyVectorEnv([lambda: SymbolicObsWrapperFullFlattenDirect(gym.make(task)) for _ in range(train_num)])
test_envs = ts.env.DummyVectorEnv([lambda: SymbolicObsWrapperFullFlattenDirect(gym.make(task)) for _ in range(test_num)])

from tianshou.utils.net.common import Net

env = gym.make(task)
state_shape = 9*9*3 + 1 # handcrafted
action_shape = env.action_space.shape or env.action_space.n
net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128])
optim = torch.optim.Adam(net.parameters(), lr=lr)

policy = ts.policy.DQNPolicy(net, optim, gamma, n_step, target_update_freq=target_freq)
train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, train_num), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)  # because DQN uses epsilon-greedy method

result = ts.trainer.offpolicy_trainer(
    policy, train_collector, test_collector, epoch, step_per_epoch, step_per_collect,
    test_num, batch_size, update_per_step=1 / step_per_collect,
    train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
    test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
    stop_fn=lambda mean_rewards: mean_rewards >= 1, # hand-crafted
    logger=logger)
print(f'Finished training! Use {result["duration"]}')

torch.save(policy.state_dict(), 'dqn.pth')
policy.load_state_dict(torch.load('dqn.pth'))