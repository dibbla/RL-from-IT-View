import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from replay_buffer import ReplayBuffer
from env import singleEnv
from torch.utils.tensorboard import SummaryWriter
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--log_csv',action='store_true',help='log csv file or not')
argparser.add_argument('--whole_buffer',action='store_true',help='sample whole batch or not')
args = argparser.parse_args()

class Q(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Q, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim,action_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Q(state_dim, hidden_dim,
                          self.action_dim).to(device)

        self.target_q_net = Q(state_dim, hidden_dim,
                                 self.action_dim).to(device)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon  # epsilon-greedy
        self.target_update = target_update # target Q update frequency
        self.count = 0
        self.device = device

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        # get data from buffer
        b_s = transition_dict['states']
        b_a = transition_dict['actions']
        b_r = transition_dict['rewards']
        b_ns = transition_dict['next_states']
        b_d = transition_dict['dones']

        # generate update data
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # compute current q values
        q_values = self.q_net(states).gather(1, actions)

        # compute target q values
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        # for logging batch q_target
        b_q = q_targets.detach().cpu().numpy()

        if args.log_csv:
            df = pd.DataFrame(list(zip(b_s, b_a, b_ns, b_d, b_r, b_q)), 
                            columns=['state', 'action', 'next_state', 'done', 'reward', 'q_target'])
            df.to_csv(f'{self.count}.csv', index=False)

        # q-loss
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))

        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        # if accumulate enough steps, update target
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())

        # count one update
        self.count += 1

        return dqn_loss

if __name__ == '__main__':
    # set up training
    lr = 2e-3
    num_episodes = 500 # totol episode for training
    hidden_dim = 48
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 16
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # add logger
    writer = SummaryWriter()

    # seeding
    env = singleEnv()
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device)

    return_list = []

    # two layer loop for visualization
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward

                    if replay_buffer.size() > minimal_size: # if buffer_size > mini_batch_size, update
                        # b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        if args.whole_buffer:
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample_whole()
                        else:
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        loss = agent.update(transition_dict)
                        writer.add_scalar('Loss/update_dqn_loss', loss, agent.count)

                return_list.append(episode_return)
                writer.add_scalar('Return/episode_return', episode_return, i_episode)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)