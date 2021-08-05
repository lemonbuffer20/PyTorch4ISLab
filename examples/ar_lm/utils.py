from typing import Union
import math
import torch


def to_ppl(loss: Union[float, torch.Tensor]) -> float:
    if isinstance(loss, torch.Tensor):
        loss = loss.item()
    return math.exp(loss)
