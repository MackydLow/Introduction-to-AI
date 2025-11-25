# Inspired from https://www.datacamp.com/tutorial/policy-gradient-theorem

import torch.nn as nn
import torch.nn.functional as nnf


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = nnf.relu(x)
        x = self.layer2(x)
        x = nnf.softmax(x, dim = -1)
        return x

