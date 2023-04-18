# Train an agent in a behavior-clone (supervised-learning) style
import pandas as pd
import torch
from torch.utils.data import Dataset

class ExpertSet(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        state = torch.tensor(self.data.iloc[idx, :-1].values, dtype=torch.float32)
        action = torch.tensor(self.data.iloc[idx, -1], dtype=torch.long)
        return state, action

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
        return self.model(x)

# Create a dataloader from the ExpertSet dataset
expert_dataset = ExpertSet('expert_data.csv')
expert_dataloader = torch.utils.data.DataLoader(expert_dataset, batch_size=32, shuffle=True)