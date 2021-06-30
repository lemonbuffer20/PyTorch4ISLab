# Day 4: Prepare ResNet

## What to reproduce:

* `my_nn/conv.py`
* `my_nn/batchnorm.py`
* `my_nn/dropout.py`
* `my_nn/pooling.py`
* `my_optim/adam.py`
* `my_optim/build.py` - add adam
* `my_model/resnet_block.py`
* `my_model/resnet20.py`
* `day04/check_resnet.py`

## Key

1. Understand **ALL** arguments in Conv2d.
2. Understand **ALL** arguments in BatchNorm2d.
3. You should remember ResNet architecture.

## ResNet

1. ResNet is fully convolutional.
2. BatchNorm (BN) and Residual connection are two core components.
3. ResNet for CIFAR and ImageNet are slightly different.

## Adam vs SGD

1. For small datasets, SGD is better. (CIFAR-10, Small language model, ...)
2. For large & complex dataset, Adam is better. (Speech, NLP, Translation, ...)
3. Adam often use 1/10 ~ 1/20 LR than SGD.