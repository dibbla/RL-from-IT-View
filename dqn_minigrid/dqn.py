import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import random
import gymnasium as gym
import numpy as np
import collections
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from replay_buffer import ReplayBuffer
import minigrid
from minigrid.wrappers import FlatObsWrapper, SymbolicObsWrapper, FullyObsWrapper
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--episodes',type=int,default=700,help='number of epochs to train')
argparser.add_argument('--log_csv',action='store_true',help='log csv file or not')
argparser.add_argument('--whole_buffer',action='store_true',help='sample whole batch or not')
argparser.add_argument('--eval_other_env',action='store_true',help='evaluate or not')
argparser.add_argument('--eval_same_env',action='store_true',help='evaluate or not')
args = argparser.parse_args()

print(
    f"settings: log_csv={args.log_csv}, whole_buffer={args.whole_buffer}, eval_other_env={args.eval_other_env}, eval_same_env={args.eval_same_env}"
    )

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
    
    def take_action_deterministic(self, state):
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
    for hidden in [8]:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>hidden:{hidden}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        # set up training
        lr = 2e-3
        num_episodes = args.episodes # totol episode for training
        hidden_dim = hidden
        gamma = 0.98
        epsilon = 0.05
        target_update = 10
        buffer_size = 10000
        minimal_size = 128
        batch_size = 128
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # create a directory for logging
        directory = f'logs/swipe-log-' +time.strftime("%Y%m%d-%H%M%S") + f'hidden-{hidden_dim}' 
        if not os.path.exists(directory):
            os.makedirs(directory)
        # create txt file for logging returns
        f = open(directory + '/returns.txt', 'w')
        # create txt file for logging losses
        f2 = open(directory + '/losses.txt', 'w')
        # create txt file for logging evaluation returns
        f3 = open(directory + '/eval_returns.txt', 'w')
        # create txt file for logging same env returns
        f4 = open(directory + '/same_env_returns.txt', 'w')
        # add one txt file tracing buffer size
        f5 = open(directory + '/buffer_size.txt', 'w')

        env = gym.make('MiniGrid-Empty-Random-6x6-v0')

        # seeding
        train_seed = 3 # this fix the agent to the upper left corner
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        # set up env & agent
        env = FlatObsWrapper(FullyObsWrapper(env))
        replay_buffer = ReplayBuffer(buffer_size)
        state, _ = env.reset(seed=train_seed) # reset for getting state_dim
        state_dim = state.shape[0]
        action_dim = env.action_space.n
        agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                    target_update, device)
        print(f'Agent setting: state_dim={state_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}')

        return_list = []
        same_env_return_list = []
        eval_return_list = []

        # training
        pbar = tqdm.tqdm(total=args.episodes, desc='Training', leave=True)
        for i in range(args.episodes):
            episode_return = 0
            state, _ = env.reset(seed=train_seed)
            done = False

            # start an episode
            while not done:
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                # env.render()
                done = terminated or truncated
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward

                if replay_buffer.size() > minimal_size: # if buffer_size > mini_batch_size, update
                    if args.whole_buffer:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample_whole()
                    else:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'rewards': b_r,
                        'next_states': b_ns,
                        'dones': b_d
                    }
                    loss = agent.update(transition_dict)
                    f2.write(str(loss.item()) + '\n')

            # dump transition_dict to .csv every 5 episodes
            if i % 5 == 0 and args.log_csv:
                df = pd.DataFrame(list(zip(b_s, b_a, b_ns, b_d, b_r)), 
                                columns=['state', 'action', 'next_state', 'done', 'reward'])
                # create csv directory
                if not os.path.exists(directory + '/csv'):
                    os.makedirs(directory + '/csv')
                df.to_csv(directory + '/csv/' + str(i) + '.csv', index=False)

            # evaluate every 5 episodes on same envs
            if i % 5 == 0 and args.eval_same_env:
                eval_done = False
                eval_return = 0
                eval_state, _ = env.reset(seed=train_seed) # same seed for evaluation
                while not eval_done:
                    action = agent.take_action_deterministic(eval_state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    eval_done = terminated or truncated
                    eval_return += reward
                    eval_state = next_state
                # print(f'Evaluation return: {eval_return}')
                f4.write(str(eval_return) + '\n')
                same_env_return_list.append(eval_return)

            # evaluate every 5 episodes on other envs
            if i % 5 == 0 and args.eval_other_env:
                avg_return = 0
                for seed in [0,1,2]:
                    eval_done = False
                    eval_return = 0
                    eval_state, _ = env.reset(seed=seed)
                    while not eval_done:
                        action = agent.take_action_deterministic(eval_state)
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        eval_done = terminated or truncated
                        eval_return += reward
                        eval_state = next_state
                    # print(f'Evaluation return: {eval_return}')
                    avg_return += eval_return
                avg_return /= 5
                f3.write(str(avg_return) + '\n')
                eval_return_list.append(avg_return)
            
            # log buffer size
            f5.write(str(replay_buffer.size()) + '\n')

            pbar.write(f'Return:{episode_return}')
            pbar.update(1)
            return_list.append(episode_return)
            f.write(str(episode_return) + '\n')
        
        print('Training finished.')
        print('Logging...')
        
        # smooth return curve by averaging with window size 5
        return_list = np.array(return_list)
        return_list = np.convolve(return_list, np.ones(5), 'valid') / 5

        same_env_return_list = np.array(same_env_return_list)
        same_env_return_list = np.convolve(same_env_return_list, np.ones(5), 'valid') / 5

        eval_return_list = np.array(eval_return_list)
        eval_return_list = np.convolve(eval_return_list, np.ones(5), 'valid') / 5

        # draw 3 separate plots
        plt.figure()
        plt.plot(return_list)
        plt.title('Return')
        plt.savefig(directory + '/return.png')

        plt.figure()
        plt.plot(same_env_return_list)
        plt.title('Same Env Return')
        plt.savefig(directory + '/same_env_return.png')

        plt.figure()
        plt.plot(eval_return_list)
        plt.title('Eval Return')
        plt.savefig(directory + '/eval_return.png')

