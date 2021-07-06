import torch
from torch.optim.optimizer import Optimizer


class MySGD(Optimizer):
    """optim.SGD"""

    def __init__(self,
                 params,
                 lr: float,
                 momentum: float = 0.0,
                 weight_decay: float = 0.0,
                 nesterov: bool = False):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]

            # ---------------------------------------------------------------- #
            # dW = -lr * grad

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                # -------------------------------- #
                # Weight decay
                # -------------------------------- #
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                # -------------------------------- #
                # Gradient momentum
                # -------------------------------- #
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                # -------------------------------- #
                # Update
                # -------------------------------- #
                p.add_(d_p, alpha=-lr)

        return loss
