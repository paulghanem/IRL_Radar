import torch.nn as nn
import torch

class CostNN(nn.Module):
    def __init__(
        self, 
        state_dim,
        hidden_dim1 = 128, 
        out_features = 1, 
    ):
        super(CostNN, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(state_dim, hidden_dim1),
            nn.LogSigmoid(),
            nn.Linear(hidden_dim1, out_features),
        )
    def forward(self, x):
        return self.net(x)        