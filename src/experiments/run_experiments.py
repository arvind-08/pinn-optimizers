import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import os
import time

from src.models.pinn_model import PINN, burgers_residual, generate_training_data

# ============================================================
# 1️⃣  Configuration
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

results_dir = "pinn_results"
os.makedirs(results_dir, exist_ok=True)

optimizers_to_run = {
    "Adam": lambda params, lr: torch.optim.Adam(params, lr=lr),
    "SGD": lambda params, lr: torch.optim.SGD(params, lr=lr),
    "RMSProp": lambda params, lr: torch.optim.RMSprop(params, lr=lr),
    "Momentum": lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.9),
}

num_epochs = 1000
learning_rate = 1e-3
n_samples = 1000

# ============================================================
# 2️⃣  Training Function
# ============================================================
def train_pinn(optimizer_name):
    model = PINN().to(device)
    x, t, u0 = generate_training_data(n_samples, device)
    optimizer = optimizers_to_run[optimizer_name](model.parameters(), lr=learning_rate)

    loss_history = []
    start_time = time.time()

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
            print(f"[{optimizer_name}] Epoch {epoch}/{num_epochs} | Loss: {loss.item():.6f}")

    duration = time.time() - start_time

    # Save loss curve
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title(f"{optimizer_name} Training Loss")
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"loss_{optimizer_name.lower()}.png"))
    plt.close()

    # Save results to JSON
    results = {
        "optimizer": optimizer_name,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "final_loss": loss_history[-1],
        "training_time_sec": round(duration, 2),
        "device": str(device),
    }
    with open(os.path.join(results_dir, f"results_{optimizer_name}.json"), "w") as f:
        json.dump(results, f, indent=2)

    return loss_history

# ============================================================
# 3️⃣  Run Experiments
# ============================================================
all_losses = {}

for opt_name in optimizers_to_run.keys():
    print(f"\n🚀 Starting experiment: {opt_name}")
    losses = train_pinn(opt_name)
    all_losses[opt_name] = losses

# ============================================================
# 4️⃣  Combined Plot for Comparison
# ============================================================
plt.figure(figsize=(8, 5))
for opt_name, losses in all_losses.items():
    plt.plot(losses, label=opt_name)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("PINN Training Loss Comparison (First-Order Optimizers)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, "loss_comparison_all.png"))
plt.close()

print("\n✅ All experiments completed successfully!")
print(f"Results saved in: {results_dir}/")
