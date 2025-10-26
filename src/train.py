# src/train.py
import torch
from tqdm import trange
import matplotlib.pyplot as plt
import os
from .model import PINN
from .data import generate_burgers_data
from .utils import pinn_total_loss

def train(layers=[2,50,50,50,1],
          optimizer_name='Adam',
          lr=1e-3,
          num_epochs=5000,
          device='cpu',
          save_dir='results/run'):
    os.makedirs(save_dir, exist_ok=True)
    model = PINN(layers).to(device)
    data = generate_burgers_data(device=device)

    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.9)
    else:
        raise ValueError("Unsupported optimizer")

    losses = []
    for epoch in trange(num_epochs):
        optimizer.zero_grad()
        total_loss, comps = pinn_total_loss(model, data)
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: total={total_loss.item():.6e}, "
                  f"ic={comps['loss_ic']:.3e}, bc={comps['loss_bc']:.3e}, f={comps['loss_f']:.3e}")
    # save model and losses
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
    torch.save({'losses': losses}, os.path.join(save_dir, 'training.pt'))

    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title(f'{optimizer_name} lr={lr}')
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()
    return model, losses
