import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

# ============================================================
#  PINN Model for Burgers' Equation (Reusable Class)
# ============================================================

class PINN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=50, hidden_layers=2, output_dim=1):
        super(PINN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        """Forward pass through the network."""
        X = torch.cat((x, t), dim=1)
        return self.net(X)

# ============================================================
#  Burgers’ Equation Residual Function
# ============================================================
def burgers_residual(model, x, t, nu=0.01 / np.pi):
    """Compute the PDE residual for the 1D Burgers’ equation."""
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)

    # Compute partial derivatives
    u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    # Residual of Burgers' equation
    f = u_t + u * u_x - nu * u_xx
    return f

# ============================================================
#  Utility Function to Generate Training Data
# ============================================================
def generate_training_data(n_samples=1000, device="cpu"):
    """Generate random (x, t) points and initial condition u(x,0)."""
    x = torch.rand(n_samples, 1) * 2 - 1  # x in [-1, 1]
    t = torch.rand(n_samples, 1)          # t in [0, 1]
    u0 = -torch.sin(np.pi * x)
    return x.to(device), t.to(device), u0.to(device)
