import torch
import torch.nn as nn
import torch.nn.functional as nnf


class PolicyModel(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim, dropout):
        super().__init__()
        kernelSize = 3
        padding = 1

        self.convol = nn.Sequential(
            nn.Conv2d(1, 16, kernelSize, padding),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernelSize, padding),
            nn.ReLU(),
            nn.Flatten()
        )

        self.nnfc1 = nn.LazyLinear(hiddenDim)
        self.nnfc2 = nn.Linear(hiddenDim, outputDim)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, i):
        i = i.permute(0, 3, 1, 2)

        i.self.convol(i)
        i.F.relu(self.nnfc1(i))
        i.self.dropout(i)
        i.self.nnfc2(i)

        return self.softmax(i)