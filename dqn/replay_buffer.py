import collections
import random
import numpy as np
import pandas as pd

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def dump(self, filename):
        """
        Dump the buffer into .csv file so brainome can be used
        """
        df = pd.DataFrame(list(self.buffer), columns=['state', 'action', 'reward', 'next_state', 'done'])
        df.to_csv(filename, index=False)

    def size(self):
        return len(self.buffer)