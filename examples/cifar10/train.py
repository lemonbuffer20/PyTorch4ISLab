import argparse
import os
import json
import wandb
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import (Compose, RandomHorizontalFlip, RandomCrop, RandomErasing,
                                    Normalize, ToTensor)

from torch4is.my_model.resnet20 import ResNet20
from torch4is.my_data.cifar10 import CIFAR10
from torch4is.my_loss.accuracy import Accuracy
from torch4is.my_optim.build import build_optimizer
from torch4is.my_optim.my_sched.build import build_scheduler
from torch4is.utils import time_log, wandb_setup


class CompositeModel(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.net = ResNet20(**cfg)
        self.loss = nn.CrossEntropyLoss()
        self.acc = Accuracy()

    def forward(self, image: torch.Tensor, label: torch.Tensor) -> dict:
        output = {}

        predict = self.net(image)
        loss = self.loss(predict, label)
        acc = self.acc(predict, label)

        output["loss"] = loss
        output["acc"] = acc
        return output


def build_dataloader(cfg: dict, dataset):
    dataloader = DataLoader(dataset, **cfg)
    return dataloader


def build_transform(mode: str = "train"):
    if mode == "train":
        transforms = Compose([
            RandomCrop(size=32, padding=2),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),  # automatically convert HWC uint8 to CHW [0, 1]
            Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            RandomErasing(p=1.0, scale=(0.125, 0.5)),
        ])
    else:  # test
        transforms = Compose([
            ToTensor(),
            Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
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
        train_acc = 0
        train_count = 0
        for count, (data, label) in enumerate(train_dataloader):
            data = data.cuda()
            label = label.cuda()
            batch_size = label.shape[0]

            optimizer.zero_grad(set_to_none=True)

            output = model(data, label)
            loss = output["loss"]
            acc = output["acc"]
            train_loss += loss.item() * batch_size
            train_acc += acc.item() * batch_size
            train_count += batch_size

            loss.backward()
            optimizer.step()

            if (count % print_interval == 0) and (count > 0):
                print(time_log())
                s = f"... train iter {count} / {len(train_dataloader)}\n" \
                    f"...... loss(now/avg): {loss.item():.6f} / {train_loss / train_count:.6f}\n" \
                    f"...... acc(now/avg): {acc.item():.4f} / {train_acc / train_count:.4f}"
                print(s)
                wandb.log({
                    "train_loss": loss.item(),
                    "train_acc": acc.item(),
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
        test_acc = 0
        test_count = 0
        with torch.no_grad():
            for count, (data, label) in enumerate(test_dataloader):
                data = data.cuda()
                label = label.cuda()
                batch_size = label.shape[0]

                output = model(data, label)
                loss = output["loss"]
                acc = output["acc"]
                test_loss += loss.item() * batch_size
                test_acc += acc.item() * batch_size
                test_count += batch_size

        test_loss /= test_count
        test_acc /= test_count
        s = f"... loss(avg): {test_loss:.6f}\n" \
            f"... acc(avg): {test_acc:.4f}"
        print(s)
        wandb.log({
            "test_loss": test_loss,
            "test_acc": test_acc,
            "iteration": iteration,
            "epoch": epoch,
        })

        scheduler.step()
        is_updated = scheduler.update_best(test_acc)
        if is_updated:
            torch.save(model.state_dict(), os.path.join(save_dir, "best.ckpt"))
            print("Best saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Configuration JSON file path.", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    train(config)
