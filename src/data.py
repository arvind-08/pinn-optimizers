# src/data.py
import torch
import numpy as np

def generate_burgers_data(N_ic=100, N_bc=100, N_f=10000, device='cpu'):
    # domain x in [-1,1], t in [0,1]
    x_ic = torch.linspace(-1, 1, N_ic, device=device).view(-1,1)
    t_ic = torch.zeros_like(x_ic)
    u_ic = -torch.sin(np.pi * x_ic)  # initial condition

    t_bc = torch.linspace(0, 1, N_bc, device=device).view(-1,1)
    x_bc_left = -torch.ones_like(t_bc)
    x_bc_right = torch.ones_like(t_bc)
    u_bc_left = torch.zeros_like(t_bc)
    u_bc_right = torch.zeros_like(t_bc)

    # collocation points (random)
    x_f = -1 + 2 * torch.rand((N_f,1), device=device)
    t_f = torch.rand((N_f,1), device=device)

    return {
        "x_ic": x_ic, "t_ic": t_ic, "u_ic": u_ic,
        "x_bc": torch.cat([x_bc_left, x_bc_right], dim=0),
        "t_bc": torch.cat([t_bc, t_bc], dim=0),
        "u_bc": torch.cat([u_bc_left, u_bc_right], dim=0),
        "x_f": x_f, "t_f": t_f
    }
