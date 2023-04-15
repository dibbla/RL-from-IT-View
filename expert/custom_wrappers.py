import gymnasium as gym
import numpy as np
from gymnasium import logger, spaces
from gymnasium.core import ObservationWrapper, ObsType, Wrapper
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
from minigrid.core.world_object import Goal

class SymbolicObsWrapperFlatten(ObservationWrapper):
    """
    Fully observable grid with a symbolic state representation.
    The symbol is a triple of (X, Y, IDX), where X and Y are
    the coordinates on the grid, and IDX is the id of the object.

    Return: full observation flattened into single vector
    """

    def __init__(self, env):
        super().__init__(env)

        new_image_space = spaces.Box(
            low=0,
            high=max(OBJECT_TO_IDX.values()),
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        objects = np.array(
            [OBJECT_TO_IDX[o.type] if o is not None else -1 for o in self.grid.grid]
        )
        agent_pos = self.env.agent_pos
        ncol, nrow = self.width, self.height
        grid = np.mgrid[:ncol, :nrow]
        _objects = np.transpose(objects.reshape(1, nrow, ncol), (0, 2, 1))

        grid = np.concatenate([grid, _objects])
        grid = np.transpose(grid, (1, 2, 0))
        grid[agent_pos[0], agent_pos[1], 2] = OBJECT_TO_IDX["agent"]
        obs["image"] = grid

        grid_id = np.zeros((ncol, nrow))
        for i in range(ncol):
            for j in range(nrow):
                grid_id[i,j] = grid[i,j][2]
        return grid_id.flatten()

class SymbolicObsWrapperIDs(ObservationWrapper):
    """
    Fully observable grid with a symbolic state representation.
    The symbol is a triple of (X, Y, IDX), where X and Y are
    the coordinates on the grid, and IDX is the id of the object.

    Return: full observation, in the same shape as grid itself. Entry contains the item
    """

    def __init__(self, env):
        super().__init__(env)

        new_image_space = spaces.Box(
            low=0,
            high=max(OBJECT_TO_IDX.values()),
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        objects = np.array(
            [OBJECT_TO_IDX[o.type] if o is not None else -1 for o in self.grid.grid]
        )
        agent_pos = self.env.agent_pos
        ncol, nrow = self.width, self.height
        grid = np.mgrid[:ncol, :nrow]
        _objects = np.transpose(objects.reshape(1, nrow, ncol), (0, 2, 1))

        grid = np.concatenate([grid, _objects])
        grid = np.transpose(grid, (1, 2, 0))
        grid[agent_pos[0], agent_pos[1], 2] = OBJECT_TO_IDX["agent"]
        obs["image"] = grid

        grid_id = np.zeros((ncol, nrow))
        for i in range(ncol):
            for j in range(nrow):
                grid_id[i,j] = grid[i,j][2]
        return grid_id