# src/tune_optuna.py
import optuna
import torch
from src.train import train  # adapt train to accept fewer args or create small training inside function

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    opt_name = trial.suggest_categorical('opt', ['Adam', 'SGD', 'momentum', 'RMSprop'])
    # do shorter training for speed, e.g., 1000 epochs
    model, losses = train(optimizer_name=opt_name, lr=lr, num_epochs=1000, device='cpu', save_dir=f'results/optuna/{opt_name}_{lr:.0e}')
    return losses[-1]

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
print(study.best_params)
