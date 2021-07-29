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
from torch4is.my_loss.gaussian_kldiv import GaussianKLDiv
from torch4is.utils import time_log, wandb_setup

from .model import MyVAE


class CompositeModel(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.beta = cfg.pop("beta")
        self.beta_warmup = cfg.pop("beta_warmup")
        self._step = 0

        self.net = MyVAE(**cfg)

        self.latent_size = self.net.latent_size

        self.recon_loss = nn.BCELoss()
        self.kl_loss = GaussianKLDiv()

    def update_beta(self):
        self._step += 1

    def forward(self, image: torch.Tensor) -> dict:
        output = {}

        predict, mu, logvar = self.net(image)
        output["mu"] = mu
        output["logvar"] = logvar

        recon_loss = self.recon_loss(predict, image)
        kl_loss = self.kl_loss(mu, logvar)

        # warmup beta
        if self._step < self.beta_warmup:
            beta = self.beta * (self._step + 1) / self.beta_warmup
        else:
            beta = self.beta

        output["loss"] = recon_loss + beta * kl_loss
        output["recon_loss"] = recon_loss
        output["kl_loss"] = kl_loss
        output["predict"] = predict
        return output

    def generate(self, z: torch.Tensor):
        return self.net.generate(z)


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
        train_recon_loss = 0
        train_kl_loss = 0
        train_count = 0
        for count, (data, _) in enumerate(train_dataloader):
            data = data.cuda()
            batch_size = data.shape[0]

            optimizer.zero_grad(set_to_none=True)

            output = model(data)
            loss = output["loss"]
            recon_loss = output["recon_loss"]
            kl_loss = output["kl_loss"]
            train_loss += loss.item() * batch_size
            train_recon_loss += recon_loss.item() * batch_size
            train_kl_loss += kl_loss.item() * batch_size
            train_count += batch_size

            loss.backward()
            optimizer.step()

            if (count % print_interval == 0) and (count > 0):
                print(time_log())
                s = f"... train iter {count} / {len(train_dataloader)}\n" \
                    f"...... loss(now/avg): {loss.item():.6f} / {train_loss / train_count:.6f}\n" \
                    f"...... recon_loss(now/avg): {recon_loss.item():.6f} / {train_recon_loss / train_count:.6f}\n" \
                    f"...... kl_loss(now/avg): {kl_loss.item():.6f} / {train_kl_loss / train_count:.6f}"
                print(s)
                wandb.log({
                    "train_loss": loss.item(),
                    "train_recon_loss": recon_loss.item(),
                    "train_kl_loss": kl_loss.item(),
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
        test_recon_loss = 0
        test_kl_loss = 0
        test_count = 0
        with torch.no_grad():
            for count, (data, _) in enumerate(test_dataloader):
                data = data.cuda()
                batch_size = data.shape[0]

                output = model(data)
                loss = output["loss"]
                recon_loss = output["recon_loss"]
                kl_loss = output["kl_loss"]
                test_loss += loss.item() * batch_size
                test_recon_loss += recon_loss.item() * batch_size
                test_kl_loss += kl_loss.item() * batch_size
                test_count += batch_size

        test_loss /= test_count
        test_recon_loss /= test_count
        test_kl_loss /= test_count
        s = f"... loss(avg): {test_loss:.6f}\n" \
            f"... recon_loss(avg): {test_recon_loss:.6f}\n" \
            f"... kl_loss (avg): {test_kl_loss:6f}"
        print(s)
        wandb.log({
            "test_loss": test_loss,
            "test_recon_loss": test_loss,
            "test_kl_loss": test_loss,
            "iteration": iteration,
            "epoch": epoch,
        })

        scheduler.step()
        is_updated = scheduler.update_best(test_loss)
        if is_updated:
            torch.save(model.state_dict(), os.path.join(save_dir, "best.ckpt"))
            print("Best saved.")

        model.update_beta()

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
                        data_np = data_np.reshape(16, 16, 3, 32, 32).transpose(0, 3, 1, 4, 2).reshape(16 * 32, 16 * 32,
                                                                                                      3)
                        data_np = np.uint8(np.clip(data_np * 255, 0, 255))  # (16, 16, 32, 32, 3)
                        Image.fromarray(data_np).save(os.path.join(save_dir, f"epoch-{epoch}-orig.png"))

                    pred_np = output["predict"].detach().cpu().numpy()  # (256, 3, 32, 32)
                    pred_np = pred_np.reshape(16, 16, 3, 32, 32).transpose(0, 3, 1, 4, 2).reshape(16 * 32, 16 * 32, 3)
                    pred_np = np.uint8(np.clip(pred_np * 255, 0, 255))  # (16, 16, 32, 32, 3)
                    Image.fromarray(pred_np).save(os.path.join(save_dir, f"epoch-{epoch}-pred.png"))

                    z1 = MyVAE.reparameterize(output["mu"], output["logvar"])[:16]  # (16, latent_dim)
                    z2 = torch.roll(z1, 1, 0)
                    z = []
                    for j in range(16):
                        scale = float(j / 15)
                        z_candidate = z1 * scale + z2 * (1.0 - scale)
                        # optional: project to unit sphere
                        z_candidate /= torch.norm(z_candidate, dim=-1, keepdim=True)
                        z.append(z_candidate)
                    z = torch.stack(z, dim=0).view(256, model.latent_size)  # (16, 16, 256) -> (256, 256)
                    gen_np = model.generate(z).detach().cpu().numpy()  # (256, 3, 32, 32)
                    gen_np = gen_np.reshape(16, 16, 3, 32, 32).transpose(0, 3, 1, 4, 2).reshape(16 * 32, 16 * 32, 3)
                    gen_np = np.uint8(np.clip(gen_np * 255, 0, 255))  # (16, 16, 32, 32, 3)
                    Image.fromarray(gen_np).save(os.path.join(save_dir, f"epoch-{epoch}-generate.png"))

                    print("Image saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Configuration JSON file path.", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    train(config)
