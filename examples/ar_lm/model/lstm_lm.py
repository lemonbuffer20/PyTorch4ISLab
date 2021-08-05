from typing import Optional, Tuple
import math
import torch
import torch.nn as nn

LSTM_STATE_TYPE = Tuple[torch.Tensor, torch.Tensor]  # hidden, cell


class LSTMLanguageModel(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 num_layers: int,
                 hidden_dim: int,
                 drop_prob: float = 0.5,
                 *, tying: bool = True):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, bias=True, batch_first=True,
                            dropout=drop_prob, bidirectional=False)
        # LSTM dropout is applied to each layer output except last layer
        self.drop = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self._initialize_parameters()

        if tying:
            self.fc.weight = self.embedding.weight

    def _initialize_parameters(self):
        nn.init.normal_(self.embedding.weight, std=float(1 / math.sqrt(self.hidden_dim)))

    @staticmethod
    def _detach_lstm_state(state: Optional[LSTM_STATE_TYPE]) -> Optional[LSTM_STATE_TYPE]:
        if state is None:
            return None
        return state[0].detach(), state[1].detach()

    def forward(self, indices: torch.Tensor, state: Optional[LSTM_STATE_TYPE]
                ) -> Tuple[torch.Tensor, LSTM_STATE_TYPE]:
        """
        :param indices:     (batch_size, seq_length)        long
        :param state:       (num_layers, batch_size, hidden_dim), (num_layers, batch_size, hidden_dim)
        :return: 
                output:     (batch_size, seq_length, vocab_size)
                state:      (num_layers, batch_size, hidden_dim), (num_layers, batch_size, hidden_dim)
        """

        hidden = self.embedding(indices)
        hidden = self.drop(hidden)

        state = self._detach_lstm_state(state)
        hidden, state = self.lstm(hidden, state)

        hidden = self.drop(hidden)
        out = self.fc(hidden)
        return out, state
