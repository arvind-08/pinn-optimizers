import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# ============================================================
# 1️⃣  Configuration
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparameters
lr = 1e-3
num_epochs = 800
optimizer_name = "Adam"
results_dir = "pinn_results"
os.makedirs(results_dir, exist_ok=True)

# ============================================================
# 2️⃣  Physics-Informed Neural Network for Burgers’ Equation
# ============================================================
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x, t):
        X = torch.cat((x, t), dim=1)
        return self.hidden(X)

# PDE residual for Burgers’ equation: u_t + u*u_x - ν*u_xx = 0
def burgers_residual(model, x, t, nu=0.01/np.pi):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)
    u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    f = u_t + u * u_x - nu * u_xx
    return f

# ============================================================
# 3️⃣  Training Data
# ============================================================
# 1000 random points in domain x ∈ [-1,1], t ∈ [0,1]
x = torch.rand(1000, 1) * 2 - 1
t = torch.rand(1000, 1)
x, t = x.to(device), t.to(device)

# Boundary condition: u(x,0) = -sin(pi * x)
u0 = -torch.sin(np.pi * x)

# ============================================================
# 4️⃣  Initialize Model + Optimizer
# ============================================================
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ============================================================
# 5️⃣  Training Loop
# ============================================================
loss_history = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    f = burgers_residual(model, x, t)
    u_pred = model(x, torch.zeros_like(t))
    loss_ic = torch.mean((u_pred - u0) ** 2)
    loss_f = torch.mean(f ** 2)
    loss = loss_ic + loss_f
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 100 == 0 or epoch == num_epochs - 1:
        print(f"[{optimizer_name}] Epoch {epoch}/{num_epochs} | Total Loss: {loss.item():.6f}")

# ============================================================
# 6️⃣  Save Results
# ============================================================
# Save loss curve
plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Total Loss")
plt.title(f"Training Loss ({optimizer_name})")
plt.grid(True)
plt.savefig(os.path.join(results_dir, f"loss_{optimizer_name.lower()}.png"))
plt.close()

# Save numeric results
results = {
    "optimizer": optimizer_name,
    "learning_rate": lr,
    "num_epochs": num_epochs,
    "final_loss": loss_history[-1],
    "device": str(device)
}
with open(os.path.join(results_dir, f"results_{optimizer_name}.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Finished training with {optimizer_name}.")
print(f"Saved plot and results to: {results_dir}/")
