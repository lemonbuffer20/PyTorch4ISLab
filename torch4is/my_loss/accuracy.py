from typing import Union, Tuple
import torch
import torch.nn as nn


class Accuracy(nn.Module):

    def __init__(self, top_k: Union[int, Tuple[int, ...]] = 1) -> None:
        super().__init__()
        if isinstance(top_k, int):
            top_k = (top_k,)
        else:
            top_k = sorted(top_k)
        self.top_k = top_k

    @torch.no_grad()  # we don't need to flow gradient
    def forward(self,
                output: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        :param output:          (batch_size, num_classes)
        :param target:          (batch_size,)
        :return:
                                (1,) if top_k == 1, else (k,)   in range [0, 1]
        """
        batch_size = target.shape[0]

        if self.top_k == (1,):
            pred = torch.argmax(output, dim=1, keepdim=False)  # (n,)
            correct = torch.eq(pred, target).float().sum().div_(batch_size)  # (n,) -> (1,)
            return correct

        max_k = max(self.top_k)
        _, pred = torch.topk(output, max_k, dim=1, largest=True, sorted=True)  # (n, k)
        pred = pred.t()  # (n, k) -> (k, n)
        correct = torch.eq(pred, target.view(1, -1).expand_as(pred))  # (k, n)

        res = []
        for k in self.top_k:
            correct_k = correct[:k].reshape(-1).float().sum().div_(batch_size)  # (k, n) - > (kn,)
            res.append(correct_k)
        return torch.as_tensor(res, device=pred.device)
