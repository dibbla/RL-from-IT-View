import gym
from gym import spaces
import numpy as np
import cv2
import random
import time
from collections import deque

GOAL = 0


def collision_with_boundaries(player):
    if player[0] > 500 or player[0] < 0 or player[1] > 500 or player[1] < 0:
        return 1
    return 0


class singleEnv(gym.Env):
    def __init__(self):
        super(singleEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-500, high=500, shape=(4 + GOAL,), dtype=np.float32)

    def step(self, action):
        self.num_steps += 1
        button_direction = action
        # Change the head position based on the button direction
        if button_direction == 1:
            self.player[0] += 50
        elif button_direction == 3:
            self.player[0] -= 50
        elif button_direction == 0:
            self.player[1] += 50
        elif button_direction == 2:
            self.player[1] -= 50

        gem_reward = 0
        # Reward for mining Gem
        if self.player == self.gem_position:
            #self.player_position = [[random.randrange(1, 10) * 50, random.randrange(1, 10) * 50]]
            #self.player = self.player_position[0]
            self.score += 1
            gem_reward = 10000
            self.total_gems += 1
            self.done = True
        else:
            self.player_position.insert(0, list(self.player))
            self.player_position.pop()

        euclidean_dist = np.linalg.norm(np.array(self.player) - np.array(self.gem_position))

        self.total_reward = gem_reward - euclidean_dist

        # On collision kill the player
        if collision_with_boundaries(self.player) == 1 or self.total_gems == 10:
            self.done = True
            if collision_with_boundaries(self.player) == 1:
                self.total_reward -= 100000
        info = {}

        head_x = self.player[0]
        head_y = self.player[1]

        #gem_delta_x = self.gem_position[0] - head_x
        #gem_delta_y = self.gem_position[1] - head_y

        # create observation:
        self.prev_actions.append(action)
        observation = [head_x, head_y, self.gem_position[0], self.gem_position[1]] + list(self.prev_actions)
        observation = np.array(observation)

        return observation, self.total_reward, self.done, info

    def render(self):
        cv2.imshow('Single_Agent_PPO', self.img)
        cv2.waitKey(15)
        # Display Grid
        self.img = np.zeros((500, 500, 3), dtype='uint8')

        # Display Gem
        cv2.rectangle(self.img, (self.gem_position[0], self.gem_position[1]),
                      (self.gem_position[0] + 50, self.gem_position[1] + 50), (0, 0, 255), -1)
        # Display Player
        for position in self.player_position:
            cv2.rectangle(self.img, (position[0], position[1]), (position[0] + 50, position[1] + 50), (255, 0, 0),
                          -1)
        t_end = time.time() + 0.10
        k = -1
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(1)
            else:
                continue
    def reset(self):
        self.img = np.zeros((500, 500, 3), dtype='uint8')
        self.total_gems = 0
        self.num_steps = 0
        # Initial Player and Gem position
        self.player_position = [[random.randrange(1, 10) * 50, random.randrange(1, 10) * 50]]
        self.gem_position = [200, 200]
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.player = self.player_position[0]

        self.done = False

        head_x = self.player[0]
        head_y = self.player[1]

        gem_delta_x = self.gem_position[0] - head_x
        gem_delta_y = self.gem_position[1] - head_y

        self.prev_actions = deque(maxlen=GOAL)  # however long we aspire the snake to be
        for i in range(GOAL):
            self.prev_actions.append(-1)  # to create history

        # create observation:
        observation = [head_x, head_y, gem_delta_x, gem_delta_y] + list(self.prev_actions)
        observation = np.array(observation)

        return observation

    def _get_obs(self):
        head_x = self.player[0]
        head_y = self.player[1]

        gem_delta_x = self.gem_position[0] - head_x
        gem_delta_y = self.gem_position[1] - head_y
        return {np.array([head_x, head_y, gem_delta_x, gem_delta_y] + list(self.prev_actions))}

    def _get_info(self):
        return {}