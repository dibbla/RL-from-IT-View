import numpy as np
import pandas as pd

RENDER_SCALAR = 50
WORLD_WIDTH = 10 * RENDER_SCALAR
WORLD_HEIGHT = 10 * RENDER_SCALAR
GEM_X = 4 * RENDER_SCALAR
GEM_Y = 4 * RENDER_SCALAR

storage_array = np.zeros(shape=(10 * 10*10, 7))
columns = ["obs.no", "agent_x", "agent_y", "gem_x", "gem_y", "rewards", "action"]
curr_index = 0
reward = 1
for i in range(10):
    for x in range(0, WORLD_WIDTH, RENDER_SCALAR):
        for y in range(0, WORLD_HEIGHT, RENDER_SCALAR):
            if GEM_X != x:
                if x < GEM_X:
                    d = 1
                elif x > GEM_X:
                    d = 3
            elif GEM_Y != y:
                if y < GEM_Y:
                    d = 0
                elif y > GEM_Y:
                    d = 2
            elif GEM_Y == y and GEM_X == x:
                d = 4
            storage_array[curr_index] = [curr_index + 1, x, y, GEM_X, GEM_Y, reward, d]
            curr_index += 1

new_expert_table = pd.DataFrame(columns=columns, data=storage_array.astype(int))
filename = f"expert_policy_path({WORLD_WIDTH // RENDER_SCALAR}x{WORLD_HEIGHT // RENDER_SCALAR}).csv"

new_expert_table.to_csv(filename, index=False)
