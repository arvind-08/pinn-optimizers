# src/model.py
import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.net = nn.ModuleList()
        for i in range(len(layers)-1):
            linear = nn.Linear(layers[i], layers[i+1])
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.net.append(linear)
        self.activation = torch.tanh

    def forward(self, x, t):
        X = torch.cat([x, t], dim=1)  # shape: (N, 2)
        for layer in self.net[:-1]:
            X = self.activation(layer(X))
        return self.net[-1](X)  # shape: (N, 1)
