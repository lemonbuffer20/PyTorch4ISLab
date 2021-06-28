import math
from torch4is.my_optim.my_sched.base_scheduler import BaseScheduler


class MyCosineLRScheduler(BaseScheduler):

    def __init__(self,
                 optimizer,
                 max_steps: int,
                 warmup_steps: int,
                 min_lr: float = 1e-8,
                 mode: str = "min"):
        super().__init__(optimizer, warmup_steps, min_lr, mode)
        self.max_steps = max_steps

    def get_lr(self, initial_lr: float, param_group_index=None, **kwargs) -> float:
        if initial_lr <= self.min_lr:
            return initial_lr

        if self.num_steps < self.warmup_steps:
            lr = initial_lr * (self.num_steps + 1) / self.warmup_steps
        elif self.num_steps >= self.max_steps:
            lr = self.min_lr
        else:
            curr_iterations = self.num_steps - self.warmup_steps
            max_iterations = self.max_steps - self.warmup_steps

            lr = self.min_lr + 0.5 * (initial_lr - self.min_lr) * (
                        1 + math.cos(math.pi * curr_iterations / max_iterations))
        return lr
