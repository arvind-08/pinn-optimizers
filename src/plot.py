# src/plot.py (pseudo)
import matplotlib.pyplot as plt
import torch, numpy as np

def plot_solution(model, x_grid, t_val, true_func=None, savepath='fig.png'):
    t_grid = torch.full_like(x_grid, t_val)
    u_pred = model(x_grid, t_grid).detach().cpu().numpy()
    plt.plot(x_grid.cpu().numpy(), u_pred, label='pred')
    if true_func:
        u_true = true_func(x_grid.cpu().numpy(), t_val)
        plt.plot(x_grid.cpu().numpy(), u_true, '--', label='true')
    plt.legend()
    plt.savefig(savepath)
    plt.close()
