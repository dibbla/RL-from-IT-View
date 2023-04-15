# import gymnasium as gym
# import numpy as np
# import minigrid
# from minigrid.wrappers import FlatObsWrapper, SymbolicObsWrapper
# from minigrid.core.constants import OBJECT_TO_IDX
# from custom_wrappers import SymbolicObsWrapperFlatten

# env = gym.make("MiniGrid-SimpleCrossingS9N1-v0", render_mode='rgb_array')
# env = SymbolicObsWrapperFlatten(env)

# obs, _ = env.reset(seed=42)

# obs, rew, terminated, truncated, _ = env.step(0)
# done = terminated or truncated