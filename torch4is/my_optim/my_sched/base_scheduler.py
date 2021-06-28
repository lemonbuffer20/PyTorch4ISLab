import torch
from torch.optim import Optimizer


class BaseScheduler(object):

    def __init__(self,
                 optimizer: Optimizer,
                 warmup_steps: int = 0,
                 min_lr: float = 1e-8,
                 mode="min"):
        self.optimizer = optimizer

        for param_group in optimizer.param_groups:
            if "initial_lr" not in param_group:
                param_group.setdefault("initial_lr", param_group["lr"])

        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

        self.num_steps = -1
        self.mode = mode.lower()
        if self.mode not in ("min", "max"):
            raise ValueError(f"[ERROR:SCHED] Scheduler mode should be either min or max, got {self.mode}.")

        self.best = None
        self._patience_count = 0
        self._step_called = True

    def update_best(self, criterion_value) -> bool:
        """Update best validation criterion and return if the value is updated."""
        if self.best is None:
            self.best = criterion_value
            self._patience_count = 0
            print(f"...... best set, {self.best:.6f}")
            return True

        prev_best = self.best
        if self.mode == "max":  # larger better
            self.best = max(self.best, criterion_value)
        else:  # smaller better
            self.best = min(self.best, criterion_value)

        is_updated = (self.best == criterion_value)
        if is_updated:
            self._patience_count = 0
            print(f"...... best updated, (old/new): ({prev_best:.6f} / {self.best:.6f})")
        else:
            self._patience_count += 1
            print(f"...... best NOT updated, (old/new): ({prev_best:.6f} / {criterion_value:.6f})\n"
                  f"...... best before: {self._patience_count} checks.")
        return is_updated

    def step(self, criterion=None) -> None:
        self.num_steps += 1

        if criterion is not None:
            _ = self.update_best(criterion)

        for i, param_group in enumerate(self.optimizer.param_groups):
            group_lr = self.get_lr(param_group["initial_lr"], param_group_index=i)
            param_group["lr"] = group_lr

        self._step_called = True

    def get_lr(self, initial_lr: float, param_group_index=None, **kwargs) -> float:
        raise NotImplementedError
