import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    in_size = layer.weight.data.size()[0]
    lim = 1./np.sqrt(in_size)
    return (-lim, lim)

class Actor(nn.Module):
    
    def __init__(self, state_size, action_size, seed=0, fc1_size=128, fc2_size=128):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size=state_size
        self.action_size=action_size
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, action_size)
        self.bn1 = nn.BatchNorm1d(fc1_size)
        self.bn2 = nn.BatchNorm1d(fc2_size)
        self.reset_params()
        
    def reset_params(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        if len(state) == self.state_size:
            state = torch.unsqueeze(state, 0)
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))
    
class Critic(nn.Module):
    
    def __init__(self, state_size, action_size, seed=0, fc1_size=128, fc2_size=128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size=state_size
        self.action_size=action_size
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size+action_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, 1)
        self.bn1 = nn.BatchNorm1d(fc1_size)
        self.bn2 = nn.BatchNorm1d(fc2_size)
        self.reset_params()
        
    def reset_params(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state, action):
        x1 = F.relu(self.fc1(state))
        x1 = self.bn1(x1)
        x = torch.cat((x1, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)