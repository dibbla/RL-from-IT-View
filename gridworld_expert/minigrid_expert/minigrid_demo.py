# This file is only used for demonstrating/testing the environments and wrappers
import gymnasium as gym
import numpy as np
import minigrid
from minigrid.wrappers import FlatObsWrapper, SymbolicObsWrapper
from minigrid.core.constants import OBJECT_TO_IDX
from custom_wrappers import SymbolicObsWrapperFlatten, SymbolicObsWrapperFlattenDirect

import matplotlib.pyplot as plt

env = gym.make("MiniGrid-SimpleCrossingS9N1-v0", render_mode='rgb_array')
env = SymbolicObsWrapperFlattenDirect(env)

obs, _ = env.reset(seed=42)

obs, rew, terminated, truncated, _ = env.step(0)
done = terminated or truncated

img = env.render()
plt.imshow(img)
plt.show()
print(obs)