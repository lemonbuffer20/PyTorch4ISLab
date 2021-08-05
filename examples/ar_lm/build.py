from typing import Optional
import torch
from wrapper import LSTMLanguageModelWrapper
from dataset import PTBDataset


def build_model(vocab_size: int,
                verbose: bool = True,
                checkpoint: Optional[str] = None) -> LSTMLanguageModelWrapper:
    model = LSTMLanguageModelWrapper(vocab_size=vocab_size,
                                     num_layers=2,
                                     hidden_dim=512,
                                     drop_prob=0.25,
                                     tying=True)
    if verbose:
        count = 0
        s = "Parameters:\n"
        for param_name, param in model.model.named_parameters():
            s += f"... {param_name:<60}\t{tuple(param.shape)}\n"
            count += param.numel()
        s += f"Total parameters: {count}"
        print(s)

    if checkpoint is not None:
        print(f"Checkpoint loaded: {checkpoint}")
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))

    return model


def build_dataset(data_root: str):
    d_train = PTBDataset(data_root, mode="train")
    dictionary = d_train.dictionary
    d_valid = PTBDataset(data_root, mode="valid", dictionary=dictionary)
    d_test = PTBDataset(data_root, mode="test", dictionary=dictionary)

    print(f"PTB data loaded: train {len(d_train)}, valid {len(d_valid)}, test {len(d_test)} words")
    print(f"PTB vocabulary size: {len(dictionary)}")
    return d_train, d_valid, d_test, dictionary
