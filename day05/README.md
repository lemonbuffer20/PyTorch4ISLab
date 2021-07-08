# Day 5: Train CIFAR-10

## What to reproduce:

* `my_loss/accuracy.py`
* `examples/cifar10/config.json`
* `examples/cifar10/train.py`

## Dataset

1. Dataset should implement `__len__` and `__getitem__`.
2. `__len__` is called when `len(xx)`.
3. `__getitem__` is called when `xx[i]`, which we call `i`'th item.
    * This is where we usually apply Transform.
    * Items will be collected and converted to batch, by `collate_fn`.

## Loss & Metric

1. Cross entropy loss is the most common.
2. CIFAR reports Top-1 accuracy.
3. Actually, we should use separate validation set to test set. (we didn't)

## Optimizer

1. SGD + Momentum + weight decay
2. **HW**: Change to Adam and plot the result. You should use much smaller LR.
3. **HW**: Using SGD, set weight decay to zero and plot the result.

## Augmentation

1. Almost always use:
    * Normalization (this is NOT an augmentation.)
    * LR flip
2. Common for CIFAR:
    * Pad-and-crop
    * Cutout (RandomErase)
3. Be careful NOT to use augmentation for valid/test.

## Load and Save Checkpoint

1. We usually want to RESUME the training from where we ended.
2. Pytorch objects have `state_dict()` and `load_state_dict()` helper.
3. Checkpoint can be any format. `torch.save` & `torch.load` will save anything.
4. Don't forget to save not only model but also optimizer and scheduler (we didn't)