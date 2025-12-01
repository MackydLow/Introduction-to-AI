import torch
import torch.nn as nn
import torch.nn.functional as nnf


class PolicyModel(nn.Module):
    def __init__(self, inputChannels = 1, hiddenDim = 128, outputDim = 4, dropout = 0.1):
        super().__init__()
        kernelSize = 3
        padding = 1

        self.convol = nn.Sequential(
            nn.Conv2d(inputChannels, 16, kernelSize, padding),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernelSize, padding),
            nn.ReLU(),
            nn.Flatten
        )

        self.nnfc1 = nn.lLinear(hiddenDim)
        self.dropout = nn.DropOut(dropout)
        self.nnfc2 = nn.linear(hiddenDim, outputDim)

    def forward(self, i):
        if i.dim() == 4 and i.shape[-1] == 1:
            i = i.permute(0, 3, 1, 2)

        i = self.convol(i)
        i = self.nnfc1(i)
        i = self.dropout(i)
        i = nnf.relu(i)
        i = self.nnfc2(i)

        return nnf.softmax(i, dim=-1)