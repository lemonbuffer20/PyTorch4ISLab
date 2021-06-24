from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

from torch4is.utils import time_log
from torch4is.my_nn.linear import MyLinear


def generate_data() -> Tuple[torch.Tensor, torch.Tensor]:
    # we want to create noisy data that follows:
    # y = Ax + b

    weight = np.array([[0.5], [-1.5]])  # actual answer
    bias = np.array([-0.05])  # actual answer

    x = np.random.uniform(-1, 1, (10000, 2)).astype(np.float32)  # created 2-dim samples x 10000 (10000, 2)
    y_answer = np.dot(x, weight) + bias  # (10000, 2) x (2, 1) + (1,) = (10000, 1)
    y_noisy_answer = y_answer + 0.01 * np.random.randn(*y_answer.shape)  # blurred answer

    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y_noisy_answer)
    return x_t, y_t


def create_network() -> nn.Module:
    net = MyLinear(2, 1)

    # print initial parameters
    print(time_log())
    for param_name, param in net.named_parameters():
        print(f"Initial parameter ({param_name}): {param} (shape: {param.shape}, num_elements: {param.numel()})")
    return net


def fit(net: nn.Module,
        data_x: torch.Tensor,
        data_y: torch.Tensor,
        max_iters: int = 100,
        w_lr: float = 0.01,
        b_lr: float = 0.0001) -> None:
    print(time_log())
    for num_iter in range(max_iters):
        net.zero_grad(set_to_none=True)  # clear gradient

        pred_y = net(data_x)  # run and generate graph

        # loss = MSE loss
        loss = torch.sum(torch.square(pred_y - data_y))
        loss.backward()

        # update using gradient descent
        with torch.no_grad():
            for param in net.parameters():
                if param.grad is None:
                    continue
                if param.ndim == 1:  # bias
                    param -= b_lr * param.grad
                else:  # weight
                    param -= w_lr * param.grad

        print(f"... iter {num_iter} / {max_iters}, loss: {loss.item()}")

    # print final parameters
    print(time_log())
    for param_name, param in net.named_parameters():
        print(f"Final parameter ({param_name}): {param}")


def run():
    data_x, data_y = generate_data()
    net = create_network()

    fit(net, data_x, data_y, max_iters=100, w_lr=1e-4, b_lr=1e-6)


if __name__ == '__main__':
    run()
