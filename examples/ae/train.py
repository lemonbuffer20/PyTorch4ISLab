import argparse
import numpy as np
from PIL import Image
import os
import json
import wandb
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import (Compose, RandomHorizontalFlip, ToTensor)

from torch4is.my_data.cifar10 import CIFAR10
from torch4is.my_optim.build import build_optimizer
from torch4is.my_optim.my_sched.build import build_scheduler
from torch4is.utils import time_log, wandb_setup

from .model import MyConvAE


class CompositeModel(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.net = MyConvAE(**cfg)
        self.loss = nn.BCELoss()

    def forward(self, image: torch.Tensor) -> dict:
        output = {}

        predict, feature = self.net(image)
        loss = self.loss(predict, image)  # reconstruction

        output["loss"] = loss
        output["predict"] = predict
        output["feature"] = feature
        return output


def build_dataloader(cfg: dict, dataset):
    dataloader = DataLoader(dataset, **cfg)
    return dataloader


def build_transform(mode: str = "train"):
    if mode == "train":
        transforms = Compose([
            RandomHorizontalFlip(p=0.5),
            ToTensor(),  # automatically convert HWC uint8 to CHW [0, 1]
        ])
    else:  # test
        transforms = Compose([
            ToTensor(),
        ])
    return transforms


def train(cfg: dict):
    save_dir = wandb_setup(cfg)
    # ------------------------------------------------------------------------ #
    # Build dataset and dataloader
    # ------------------------------------------------------------------------ #
    train_dataset = CIFAR10(cfg["dataset"]["data_dir"], mode="train",
                            transform=build_transform("train"))
    train_dataloader = build_dataloader(cfg["dataloader"]["train"], train_dataset)

    test_dataset = CIFAR10(cfg["dataset"]["data_dir"], mode="test",
                           transform=build_transform("test"))
    test_dataloader = build_dataloader(cfg["dataloader"]["test"], test_dataset)

    # ------------------------------------------------------------------------ #
    # Build model
    # ------------------------------------------------------------------------ #
    model = CompositeModel(cfg["model"])
    model.cuda()

    if cfg["training"]["checkpoint"] is not None:
        model.load_state_dict(torch.load(cfg["training"]["checkpoint"], map_location="cuda"))
        print("Resume model.")

    # ------------------------------------------------------------------------ #
    # Build optimizer and scheduler
    # ------------------------------------------------------------------------ #
    optimizer = build_optimizer(cfg["optimizer"], model.parameters())
    scheduler = build_scheduler(cfg["scheduler"], optimizer)
    scheduler.step()  # we need first call to set lr

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #
    iteration = 0
    max_epochs = cfg["training"]["max_epochs"]
    print_interval = cfg["training"]["print_interval"]
    visualize_interval = cfg["training"]["visualize_interval"]
    for epoch in range(max_epochs):
        # ------------------------------------------------------------------------ #
        # Train loop
        # ------------------------------------------------------------------------ #
        print(time_log())
        print(f"Training epoch {epoch} / {max_epochs} start!")
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"... current LR: {current_lr:.6f}")
        model.train()

        train_loss = 0
        train_count = 0
        for count, (data, _) in enumerate(train_dataloader):
            data = data.cuda()
            batch_size = data.shape[0]

            optimizer.zero_grad(set_to_none=True)

            output = model(data)
            loss = output["loss"]
            train_loss += loss.item() * batch_size
            train_count += batch_size

            loss.backward()
            optimizer.step()

            if (count % print_interval == 0) and (count > 0):
                print(time_log())
                s = f"... train iter {count} / {len(train_dataloader)}\n" \
                    f"...... loss(now/avg): {loss.item():.6f} / {train_loss / train_count:.6f}"
                print(s)
                wandb.log({
                    "train_loss": loss.item(),
                    "iteration": iteration,
                    "epoch": epoch,
                    "lr": current_lr,
                })

            iteration += 1

        # ------------------------------------------------------------------------ #
        # Test loop
        # ------------------------------------------------------------------------ #
        print(time_log())
        print(f"Test epoch {epoch} / {max_epochs} start!")
        model.eval()

        test_loss = 0
        test_count = 0
        with torch.no_grad():
            for count, (data, _) in enumerate(test_dataloader):
                data = data.cuda()
                batch_size = data.shape[0]

                output = model(data)
                loss = output["loss"]
                test_loss += loss.item() * batch_size
                test_count += batch_size

        test_loss /= test_count
        s = f"... loss(avg): {test_loss:.6f}"
        print(s)
        wandb.log({
            "test_loss": test_loss,
            "iteration": iteration,
            "epoch": epoch,
        })

        scheduler.step()
        is_updated = scheduler.update_best(test_loss)
        if is_updated:
            torch.save(model.state_dict(), os.path.join(save_dir, "best.ckpt"))
            print("Best saved.")

        # ------------------------------------------------------------------------ #
        # Visualize
        # ------------------------------------------------------------------------ #
        if epoch % visualize_interval == 0:  # every 5 epochs
            with torch.no_grad():
                for count, (data, _) in enumerate(test_dataloader):
                    if count >= 1:  # only run first one
                        break
                    data = data[:256].cuda()  # 256 samples
                    output = model(data)

                    if epoch == 0:  # save original samples only at beginning
                        data_np = data.detach().cpu().numpy()  # (256, 3, 32, 32)
                        data_np = data_np.reshape(16, 16, 3, 32, 32).transpose(0, 3, 1, 4, 2).reshape(16 * 32, 16 * 32, 3)
                        data_np = np.uint8(np.clip(data_np * 255, 0, 255))  # (16, 16, 32, 32, 3)
                        Image.fromarray(data_np).save(os.path.join(save_dir, f"epoch-{epoch}-orig.png"))

                    pred_np = output["predict"].detach().cpu().numpy()  # (256, 3, 32, 32)
                    pred_np = pred_np.reshape(16, 16, 3, 32, 32).transpose(0, 3, 1, 4, 2).reshape(16 * 32, 16 * 32, 3)
                    pred_np = np.uint8(np.clip(pred_np * 255, 0, 255))  # (16, 16, 32, 32, 3)
                    Image.fromarray(pred_np).save(os.path.join(save_dir, f"epoch-{epoch}-pred.png"))
                    print("Image saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Configuration JSON file path.", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    train(config)
