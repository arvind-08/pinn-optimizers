# Hypertuning Optimization Techniques for Physics-Informed Neural Networks (PINNs)

This repository contains the implementation for my MTech thesis project titled  
**“Hypertuning the Optimization Techniques for Physics-Informed Neural Networks (PINNs)”**.

The project focuses on analyzing and comparing various **first-order** and **second-order optimization algorithms** for training Physics-Informed Neural Networks (PINNs) on benchmark PDE problems such as the **Burgers’ equation**, **Heat equation**, and **Poisson equation**.

---

## 🧠 Project Overview

Physics-Informed Neural Networks (PINNs) are a class of neural networks that integrate **physical laws** (expressed as partial differential equations) into the loss function, allowing data-efficient and physically consistent learning.

This project investigates the performance of different optimizers—both **first-order** and **second-order**—and explores **hyperparameter tuning** to achieve faster convergence, better stability, and improved generalization.

### **Implemented Optimizers**
#### 🥇 First-Order
- Stochastic Gradient Descent (SGD)
- Momentum
- RMSProp
- Adam

## ⚙️ Installation

### **1. Clone the repository**
```bash
git clone https://github.com/<your-username>/pinn-optimizers.git
cd pinn-optimizers
```
### **2. Create and activate a virtual environment**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1       # For PowerShell (Windows)
source venv/bin/activate          # For macOS/Linux
```

### **3. Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Running Experiments

Train a PINN with Adam optimizer
```bash
python -c "from src.train import train; train(optimizer_name='Adam', lr=1e-3, num_epochs=5000, device='cpu', save_dir='results/adam')"
```
Train a PINN with SGD optimizer
```bash
python -c "from src.train import train; train(optimizer_name='SGD', lr=1e-3, num_epochs=5000, device='cpu', save_dir='results/sgd')"
```
Train a PINN with RMSprop optimizer
```bash
python -c "from src.train import train; train(optimizer_name='RMSprop', lr=1e-3, num_epochs=5000, device='cpu', save_dir='results/rmsprop')"
```
Train a PINN with momentum optimizer
```bash
python -c "from src.train import train; train(optimizer_name='momentum', lr=1e-3, num_epochs=5000, device='cpu', save_dir='results/momentum')"
```
