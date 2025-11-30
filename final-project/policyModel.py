import torch.nn as nn
import torch.nn.functional as nnf

class PolicyModel(nn.Module):
    def __init__(self, inputChannels, hiddenDim, outputDim, droupout):
        super().__init__()
        inputChannels = 1
        hiddenDim = 128
        outputDim = 4
        dropout = 0.1
        kernelSize = 3
        padding = 1

        self.convol = nn.seq(
            nn.convol2D(inputChannels, 16, kernelSize, padding),
            nn.ReLU(),
            nn.convol2D(16, 32, kernelSize, padding),
            nn.ReLU(),
            nn.Flatten
        )

        self.nnfc1 = nn.lLinear(hiddenDim)
        self.dropout = nn.DropOut(dropout)
        self.nnfc2 = nn.linear(hiddenDim, outputDim)

    def forward(self, i):
        if i.dim() == 4 and i.shape[-1] == 1:
            i = i.permute(0, 3, 1, 2)

        i = self.convol(x)
        i = self.nnfc1(x)
        i = self.dropout(x)
        i = nnf.relu(x)
        i = self.nnfc2(x)

        return nnf.softmax(i, dim=-1)