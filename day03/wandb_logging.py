import argparse
import wandb
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch4is.utils import wandb_setup
from torch4is.my_optim import build_optimizer
from torch4is.my_optim.my_sched import build_scheduler


def run(cfg: dict):
    # ----------------------------------------------------------------------- #
    # WandB
    # ------------------------------------------------------------------------ #
    _ = wandb_setup(cfg)

    # ------------------------------------------------------------------------ #
    # Optimizer and Scheduler
    # ------------------------------------------------------------------------ #
    dummy_params = [nn.Parameter(torch.ones(1)), nn.Parameter(torch.zeros(1))]
    optimizer = build_optimizer(cfg["optimizer"], dummy_params)
    scheduler = build_scheduler(cfg["scheduler"], optimizer)

    # ------------------------------------------------------------------------ #
    # Loop
    # ------------------------------------------------------------------------ #
    lrs = []
    for t in range(1050):
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"LR {t}: {current_lr:.5f}")
        wandb.log({"lr": current_lr, "step": t})
        lrs.append(current_lr)

    # ------------------------------------------------------------------------ #
    # Draw
    # ------------------------------------------------------------------------ #
    plt.figure()
    plt.plot(lrs)
    plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Configuration file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        configs = json.load(f)

    run(configs)
