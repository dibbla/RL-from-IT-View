import gymnasium as gym
import numpy as np
import minigrid
from minigrid.core.constants import OBJECT_TO_IDX

print(f"{OBJECT_TO_IDX=}")

env = gym.make("MiniGrid-SimpleCrossingS9N1-v0", render_mode="rgb_array")
env = minigrid.wrappers.SymbolicObsWrapper(env)
obs, info = env.reset(seed=123)
r = env.render()
grid = obs["image"]
print(grid[np.where(grid[:, :, 2] == OBJECT_TO_IDX["goal"])]) # (7,7,8), x=7 y=7 type=8(goal)