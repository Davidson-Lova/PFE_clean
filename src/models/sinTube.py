#%%
import torch
import torch.nn as nn
import numpy as np

# %%
class resNet(nn.Module) :
    def __init__(self, nin, nout) :
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.Layer1 = nn.Linear(nin, 15)
        self.Layer2 = nn.Linear(15, 15)
        self.Layer3 = nn.Linear(15, nout)

    def forward(self, tX) :
        x = self.Layer1(tX)
        x = torch.relu(x)
        x = self.Layer2(x)
        x = torch.relu(x)
        x = self.Layer3(x)
        return x
    
    def zero_grad(self, set_to_none: bool = False) -> None:
        return super().zero_grad(set_to_none)
    
    def set_params(self, param_list):
        for p1, p2 in zip(list(self.parameters()), param_list) :
            p1.data = p2
        return self