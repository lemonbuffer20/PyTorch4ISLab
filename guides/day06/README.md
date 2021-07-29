# Day 6: Train AutoEncoder on CIFAR-10

## What to reproduce:

* `my_nn/upsample.py`
* `examples/ae/config.json`
* `examples/ae/model.py`
* `examples/ae/train.py`


## Visualize

1. Visualizing the result is really important.
2. Result includes: training curve, test performance, tempting visualization.

## Convolutional Upsampling

1. For CNNs, there are three ways to upsample the resolution.
    * Bilinear (or Bicubic) upsample
    * Nearest neighbor upsample
    * Transposed convolution
2. It is often considered that ConvTranspose may develop some artifacts.
3. All three are differentiable.

## AutoEncoder

1. The goal of AE is to extract the informative feature in a compressed form (vector)
2. Often, image takes x2 channels and 1/2 height, width.
3. How can we measure the quality? Loss is not direct measure of natural-ness.
4. Try to improve the quality to generated images:
   * **HW**: Use replication padding instead of zero padding?
   * **HW**: Use bilinear upsampling instead of the nearest neighbor?
   * **HW**: Use MSE loss instead of BCE loss?
5. Try to compress more:
   * **HW** Compress more. Use fewer channels & more down-sampling.