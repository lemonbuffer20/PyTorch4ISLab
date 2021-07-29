import torch

from torch4is.my_model.resnet20 import ResNet20


def run():
    net = ResNet20()

    dummy_input = torch.zeros(16, 3, 32, 32)

    dummy_output = net(dummy_input)
    print(f"Input: {tuple(dummy_input.shape)} -> Output: {tuple(dummy_output.shape)}")

    param_count = 0
    print("Parameters:")
    for n, p in net.named_parameters():
        print(f"... {n} ({tuple(p.shape)})")
        param_count += p.numel()
    print(f"Total parameters: {param_count}")


if __name__ == '__main__':
    run()
