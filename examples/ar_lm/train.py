from typing import Optional
import os
import math
import copy
import numpy as np
import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils import clip_grad_norm_

from build import build_model, build_dataset
from utils import to_ppl


def batchify(data: np.ndarray, batch_size: int) -> np.ndarray:
    steps_per_batch = len(data) // batch_size
    data = data[:steps_per_batch * batch_size]
    data = data.reshape(batch_size, -1)  # (b, s)
    return data


def train(data_root: str,
          device="cuda",
          verbose: bool = False,
          checkpoint_path: Optional[str] = None):
    # ---------------------------------------------------------------- #
    # Build
    print("-" * 64)
    # ---------------------------------------------------------------- #
    d_train, d_valid, _, dictionary = build_dataset(data_root)
    vocab_size = len(dictionary)

    model = build_model(vocab_size, verbose=verbose, checkpoint=checkpoint_path)
    model.to(device)

    # ---------------------------------------------------------------- #
    # Train
    print("-" * 64)
    max_epochs = 200
    batch_size = 16
    unroll_steps = 24
    valid_batch_size = 4
    # ---------------------------------------------------------------- #
    print("Train start!")
    model.train()
    # in this example, we simply train network to overfit the data
    # optimizer = SGD(model.parameters(), lr=1.0, momentum=0.9, weight_decay=0.001)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = MultiStepLR(optimizer, milestones=[6] + list(range(7, max_epochs)), gamma=0.96)

    # ---------------------------------------------------------------- #
    # Prepare valid data for all epoch
    valid_data = copy.deepcopy(d_valid.data)
    valid_data = np.array(valid_data, dtype=np.int64)
    valid_data = batchify(valid_data, valid_batch_size)
    valid_steps = valid_data.shape[1]

    best_loss = math.inf
    for epoch_idx in range(max_epochs):
        print("-" * 64)
        print(f"... Epoch {epoch_idx} start!")
        print(f"... Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        # ---------------------------------------------------------------- #
        # Prepare training data for current epoch, shuffle for every epoch
        train_data = copy.deepcopy(d_train.data)
        _shuffle_idx = np.random.randint(len(train_data))
        train_data = train_data[_shuffle_idx:] + train_data[:_shuffle_idx]
        train_data = np.array(train_data, dtype=np.int64)
        train_data = batchify(train_data, batch_size)
        train_steps = train_data.shape[1]

        model.train()
        state = None
        train_loss_sum = 0
        train_count = 0
        for count, i in enumerate(range(0, train_steps - 1, unroll_steps)):
            _end_step = min(i + unroll_steps, train_steps - 1)
            indices = train_data[:, i: _end_step]
            targets = train_data[:, i + 1:_end_step + 1]

            indices = torch.from_numpy(indices).to(device)
            targets = torch.from_numpy(targets).to(device)

            optimizer.zero_grad(set_to_none=True)
            loss, state = model(indices, targets, state)
            train_loss_sum += loss * (_end_step - i)
            train_count += (_end_step - i)

            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=0.5)

            if count % 200 == 0:
                print(f"...... [{i + 1}/{train_steps}] "
                      f"ppl: {to_ppl(loss):.3f} (avg: {to_ppl(train_loss_sum / train_count):.3f}) "
                      f"(grad_norm: {grad_norm.item():.3f})")
            optimizer.step()

        print("-" * 64)
        train_loss = train_loss_sum / train_count
        print(f"...... Train ppl: {to_ppl(train_loss):.3f}")
        # ---------------------------------------------------------------- #
        # Validate the model
        model.eval()
        state = None
        valid_loss_sum = 0
        valid_count = 0
        with torch.no_grad():
            for i in range(0, valid_steps - unroll_steps - 1, unroll_steps):
                _end_step = min(i + unroll_steps, valid_steps - 1)
                indices = valid_data[:, i: _end_step]
                targets = valid_data[:, i + 1:_end_step + 1]

                indices = torch.from_numpy(indices).to(device)
                targets = torch.from_numpy(targets).to(device)

                loss, state = model(indices, targets, state)
                valid_loss_sum += loss * (_end_step - i)
                valid_count += (_end_step - i)

        valid_loss = valid_loss_sum / valid_count
        print(f"...... Valid ppl: {to_ppl(valid_loss):.3f}")
        if valid_loss < best_loss:
            print(f"...... Valid best updated: {to_ppl(best_loss):.3f} -> {to_ppl(valid_loss):.3f}")
            best_loss = valid_loss
            torch.save(model.state_dict(), f"result/ar_lm/best.pth")
        else:
            print(f"...... Valid best NOT updated, best: {to_ppl(best_loss):.3f}")

        scheduler.step()
        torch.save(model.state_dict(), f"result/ar_lm/latest.pth")

    # ---------------------------------------------------------------- #
    print("-" * 64)
    print("Train Done!")


if __name__ == '__main__':
    os.makedirs("result/ar_lm", exist_ok=True)

    train(data_root="ptb", device="cuda", verbose=True)
    # train(data_root="ptb", device="cuda", verbose=True, checkpoint_path="result/ar_lm/best.pth")
