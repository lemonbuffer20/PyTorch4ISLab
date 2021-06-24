# Day 2: Pytorch Tensor, Functional, Module.

### What to reproduce:
* `torch4is/my_nn/relu.py`
* `torch4is/my_nn/leaky_relu.py`
* `torch4is/my_nn/softmax.py`
* `torch4is/my_nn/linear.py`
* `day02/linear_regression.py`

## Key

1. You should know what Tensor is.
2. You should understand the concept of .grad
3. You should be familiar with broadcast, view and contiguous.
4. You should not fear F and nn.Module.

## Tensor

1. Tensor is N-dimensional matrix-like elements, but actually is a chunk of continued values.
2. Tensor has two side: `data` and `grad`.
3. Tensor has attributes: `shape` (=size), `device`, `dtype`, `requires_grad`
4. Tensor has more attributes, which are mostly not used: `is_leaf`, `is_contiguous`, `stride`, `data_ptr`
5. We can re-arrange data by:
    * `view(...)` -> does not change internal order, require contiguous.
    * `reshape(...)` -> may not change internal order
    * `transpose(i, j)` -> change axis
    * `squeeze(i)`/`unsqueeze(i or None)` -> add/remove axis
6. We can convert data by:
    * `to(...)` -> most general, for both type and device. If changes, it will create copy.
    * `cpu()` (= `to('cpu')`) -> move tensor to CPU
    * `cuda()` (= `to('cuda')`) -> move tensor to GPU
    * `clone()` -> make same data but copied version
    * `numpy()` -> copy CPU tensor to numpy array. Almost always use as `cpu().numpy()`
    * `tolist()` -> copy CPU/GPU tensor to list in CPU.
    * `item()` -> get int/float scalar from Tensor. Only applicable for 1-element Tensor.
7. Default tensor creation setting is float32 and CPU.
8. It is better to create Tensor at the time then using `to(...)`.
```python
t1 = torch.tensor([1.0, 2.0], dtype=torch.float32, device="cuda")  # GOOD
t2 = torch.tensor([1.0, 2.0], dtype=torch.float32).to("cuda")  # NOT GOOD
```

## Tensor operations

1. Tensor has inplace operations.
2. Unless specified (`with torch.no_grad()`), tensor will automatically track gradient.
3. Some operations are not differentiable.
4. Some operations are not differentiable in inplace mode.
5. Tensor inputs should be BROADCASTABLE.
5. Most operations are in:
```python
import torch
import torch.nn.functional as F
```

> TIP: What is functional? Functional is a function that does not have any state inside. 
> The operation should be fully performed with inputs. However, we need state for NNs, for example, parameters. 

## Modules

1. Base class, which is a building block of Pytorch.
2. SHOULD-IMPLEMENT member functions: 
   * `__init__` -> called when object is created.
   * `forward` -> called inside `__call__`', which makes each module Callable.
3. MUST-KNOW member functions:
   * `named_parameters()`, `parameters()`, `named_buffers()`, `buffers()`
   * `named_modules()`, `modules()`
   * `train()`, `eval()` -> change internal flag `training` to T/F, recursively.
   * `state_dict()`, `load_state_dict()`
   * `register_parameter()` -> may not need, because `nn.Parameter()` is automatically registered.
   * `register_buffer()` -> should be called manually.
4. Most modules are in:
```python
import torch
import torch.nn as nn
```
5. Both Parameters and Buffers and states (= saved).
   * Parameters are basically leaf Tensors with requires_grad=True.
   * Buffers are basically helpers, with requires_grad=False,
   * For multi-GPU case, changes are shared through GPUs for Parameters but not for Buffers.

## AutoGrad

1. Each tensor has its own gradient in `.grad`, which is also a Tensor.
2. We need to track the history (if needed), which needs GPU memory.
3. Gradient is actually calculated when `t.backward()` is called.
   * When we call `backward()`, the history of non-leaf tensors are all gone, unless specified by `retain_grad`.
   * `backward()` can get gradient as input argument.
   * Only leaf Tensors(ex: parameters) with `requires_grad=True` will keep gradient. 
3. We need special context NOT TO track the gradient.
```python
with torch.no_grad():
    ...
```
> TIP: Tensorflow takes opposite; they need special context TO track the gradient.
4. Gradient can be cleaned up, by setting it to None.
```python
t.grad.zero_()  # NOT BAD
t.grad = None  # GOOD

opt.zero_grad()  # NOT BAD
opt.zero_grad(set_to_none=True)  # GOOD
```
5. If we don't clean up and call backward multiple times, gradient will be accumulated.
   * Used to expand batch size with limited GPU resources.

