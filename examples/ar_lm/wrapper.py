from typing import Optional, Tuple, List
# import math
import torch
import torch.nn as nn

from model.lstm_lm import LSTMLanguageModel, LSTM_STATE_TYPE


class LSTMLanguageModelWrapper(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 num_layers: int,
                 hidden_dim: int,
                 drop_prob: float = 0.5,
                 *, tying: bool = True) -> None:
        super().__init__()

        self.vocab_size = vocab_size

        self.model = LSTMLanguageModel(vocab_size, num_layers, hidden_dim, drop_prob, tying=tying)
        self.loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self,
                indices: torch.Tensor,
                targets: torch.Tensor,
                state: Optional[LSTM_STATE_TYPE]) -> Tuple[torch.Tensor, LSTM_STATE_TYPE]:
        # this function returns CE loss, so it requires target (which should be right shifted 1 step)
        output, state = self.model(indices, state)
        loss = self.loss(output.view(-1, self.vocab_size), targets.view(-1))
        return loss, state

    @torch.no_grad()
    def greedy_generate(self, indices: List[int], generate_length: int = 100) -> List[int]:
        raise NotImplementedError
