# Train an agent in a behavior-clone (supervised-learning) style
import sys

sys.path.append("..")

import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from stable_baselines3 import PPO
from env import singleEnv

EXPERT_PATH = "expert_policy.csv"

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
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_size),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        action = self.model(x)
        return action


# Training function
def train_loop(model, dataloader, val_dataloader, optimizer, criterion, device, epochs):
    # Load model and environment
    env = singleEnv()
    env.reset()
    model_path = f"final_models/380000.zip"
    expert_policy = PPO.load(model_path, env=env)
    model.train()
    for epoch in range(epochs):
        for i, (states, actions) in enumerate(dataloader):
            model.train()
            states = states.to(torch.float).to(device)
            actions = actions.to(torch.long).to(device)  # change to long
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
            if i % 100 == 0:
                # Evaluate the model on the validation set
                model.eval()
                with torch.no_grad():
                    val_loss, val_acc = 0, 0
                    for val_states, val_actions in val_dataloader:
                        val_states = val_states.to(torch.float).to(device)
                        val_actions = val_actions.to(torch.long).to(device)

                        for x in range(len(val_states)):
                            expert_action, _ = expert_policy.predict(val_states[x][:4].cpu())
                            if expert_action == val_actions[x].item():
                                val_acc += (1/len(val_states))

                        val_outputs = model(val_states)
                        val_loss += criterion(val_outputs, val_actions).item()
                    val_loss /= len(val_dataloader)
                    val_acc /= len(val_dataloader)

                print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")


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

    train_loop(bc_agent, train_loader, val_loader, optimizer, criterion, device, 10)
