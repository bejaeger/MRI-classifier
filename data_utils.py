from typing import *

import numpy as np
import torch
from torch import Tensor
from numpy.typing import NDArray
from torch.nn.functional import pad
from torch.utils.data import Dataset

class TorchDataset(Dataset):
    def __init__(self, images: Sequence[Tensor], labels: Sequence[int]) -> None:
        self.images = images
        self.labels = [torch.tensor(label) for label in labels]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        return self.images[idx], self.labels[idx]


def pad_image(image: NDArray, target_depth: int) -> Tensor:
    image = torch.from_numpy(image.astype(np.float32))
    current_depth = image.shape[-1]
    if current_depth < target_depth:
        pad_total = target_depth - current_depth
        pad_left = pad_total // 2
        pad_right = pad_total // 2
        if pad_total % 2 == 1:
            pad_right += 1
        padding = (pad_left, pad_right)  # [left, right, top, bottom, front, back]
        image = pad(image, padding, "constant", 0)
    return image


def pad_images(images: Sequence[NDArray], target_depth: int = 40) -> Sequence[Tensor]:
    padded_images = []
    for image in images:
        padded_images.append(pad_image(image=image, target_depth=target_depth))
    return padded_images


def preprocess(images: Sequence[NDArray]) -> Sequence[Tensor]:
    return pad_images(images=images, target_depth=40)
