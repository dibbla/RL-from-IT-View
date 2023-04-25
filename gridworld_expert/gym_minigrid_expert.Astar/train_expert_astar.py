import gymnasium as gym
import numpy as np
import minigrid
from minigrid.wrappers import FlatObsWrapper, SymbolicObsWrapper
from minigrid.core.constants import OBJECT_TO_IDX
from custom_wrappers import SymbolicObsWrapperFlatten, SymbolicObsWrapperIDs
from A_star import astar

env = gym.make("MiniGrid-SimpleCrossingS9N1-v0", render_mode='rgb_array')
env = SymbolicObsWrapperIDs(env)

obs, _ = env.reset(seed=42)

start = (1,1)
goal = (7,7)

path = astar(start, goal, obs)

for node, move in path:
    if move is None:
        print(node)
    else:
        print(node, move)