import pickle
import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data.dataset import Dataset


class CIFAR10(Dataset):
    """Download CIFAR-10
    https://www.cs.toronto.edu/~kriz/cifar.html (python version)
    """

    def __init__(self,
                 data_dir: str,
                 mode: str = "train",
                 transform=None):
        super().__init__()
        self.data_dir = data_dir

        if mode not in ("train", "test"):
            raise ValueError(f"Invalid mode {mode}, should be either train or test.")
        self.mode = mode
        self.transform = transform

        if mode == "train":
            batches = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
        else:
            batches = ["test_batch"]

        images = []
        labels = []
        for b in batches:
            with open(os.path.join(data_dir, b), "rb") as f:
                dump = pickle.load(f, encoding="latin1")
                images.append(dump["data"])  # (10000, 3072) numpy array
                labels.extend(dump["labels"])  # (10000,) list of int

        # by doing this way, we keep ALL data in memory.
        images = np.concatenate(images).reshape(-1, 3, 32, 32)  # uint8 CHW
        images = images.transpose(0, 2, 3, 1)  # HWC format
        labels = np.array(labels, dtype=np.int64)
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels
        print(f"CIFAR-10 {mode}, images shape: {images.shape}, labels shape: {labels.shape}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = self.images[index]
        label = self.labels[index]

        if self.transform is not None:
            image = Image.fromarray(image)  # torchvision transforms require PIL image.
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label


if __name__ == '__main__':
    d = CIFAR10("/home/khshim/project/Pytorch4ISLab/data/cifar-10", mode="train")
