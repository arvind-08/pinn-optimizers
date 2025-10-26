# src/utils.py
import torch

def gradients(u, x):
    return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

def compute_pde_residual(model, x, t, nu=0.01/torch.pi):
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = model(x, t)
    u_t = gradients(u, t)
    u_x = gradients(u, x)
    u_xx = gradients(u_x, x)
    f = u_t + u * u_x - nu * u_xx
    return f

def pinn_total_loss(model, data, lambda_f=1.0, nu=0.01/torch.pi):
    x_ic, t_ic, u_ic = data['x_ic'], data['t_ic'], data['u_ic']
    x_bc, t_bc, u_bc = data['x_bc'], data['t_bc'], data['u_bc']
    x_f, t_f = data['x_f'], data['t_f']

    u_ic_pred = model(x_ic, t_ic)
    loss_ic = torch.mean((u_ic_pred - u_ic)**2)

    u_bc_pred = model(x_bc, t_bc)
    loss_bc = torch.mean((u_bc_pred - u_bc)**2)

    f_pred = compute_pde_residual(model, x_f, t_f, nu=nu)
    loss_f = torch.mean(f_pred**2)

    total = loss_ic + loss_bc + lambda_f * loss_f
    return total, {'loss_ic': loss_ic.item(), 'loss_bc': loss_bc.item(), 'loss_f': loss_f.item()}
