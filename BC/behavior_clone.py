# Train an agent in a behavior-clone (supervised-learning) style
import sys
sys.path.append("..")

import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from env import singleEnv

EXPERT_PATH = "gridworld_expert/gym_expert/expert_policy.csv"

# For loading expert dataset
class ExpertSet(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        state = torch.tensor(self.data.iloc[idx, 1:-1].values, dtype=torch.float32)
        action = torch.tensor(self.data.iloc[idx, -1], dtype=torch.long)
        return state, action

# Agent architect
class BC(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(BC, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, output_size),
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        action =  self.model(x)
        return action

# Training function
def train_loop(model, dataloader, val_dataloader, optimizer, criterion, device, epochs):
    model.train()
    for epoch in range(epochs):
        for i, (states, actions) in enumerate(dataloader):
            model.train()
            states = states.to(torch.float).to(device)
            actions = actions.to(torch.long).to(device) # change to long
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
            if i%1000 == 0:
                # Evaluate the model on the validation set
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for val_states, val_actions in val_dataloader:
                        val_states = val_states.to(torch.float).to(device)
                        val_actions = val_actions.to(torch.long).to(device)
                        val_outputs = model(val_states)
                        val_loss += criterion(val_outputs, val_actions).item()
                    val_loss /= len(val_dataloader)
                    
                    # TODO: Create an environment to validate on reward, need a observation wrapper
                    # env = singleEnv()
                    # total_reward = 0
                    # for i in range(100):
                    #     state = env.reset()
                    #     done = False
                    #     while not done:
                    #         state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                    #         action_probs = model(state_tensor)
                    #         action = torch.argmax(action_probs).item()
                    #         next_state, reward, done, _ = env.step(action)
                    #         state = next_state
                    #         total_reward += reward
                    avg_reward = 0
                model.train()
                print(f"Validation Loss: {val_loss:.4f}, Average Reward: {avg_reward:.4f}")


if __name__ == '__main__':
    # Load the expert dataset
    expert_dataset = ExpertSet(EXPERT_PATH)
    # train-val split
    train_size = int(0.8 * len(expert_dataset))
    val_size = len(expert_dataset) - train_size

    train_dataset, val_dataset = random_split(expert_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # Prepare for training
    device = 'cuda:0'
    bc_agent = BC(input_size=5, output_size=4).to(device)
    optimizer = torch.optim.Adam(bc_agent.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_loop(bc_agent,train_loader,val_loader,optimizer,criterion,device,1)