import numpy as np
import torch
from tests import _PATH_DATA

train_images = torch.load(f"{_PATH_DATA}/processed/train_images.pt")
train_labels = torch.load(f"{_PATH_DATA}/processed/train_labels.pt")


def test_len():
    assert len(train_images) == 25000, "Length of train_images is expected to be 25000"


def test_shape():
    assert train_images.shape[1] == 28, "Height of train_images is expected to be 28"
    assert train_images.shape[2] == 28, "Width of train_images is expected to be 28"


def test_labels():
    assert len(train_labels) == 25000, "Length of train_labels is expected to be 25000"
    np.unique(train_labels.numpy()) == np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ), "10 classes are expected, equal to labels numbered from 0-9"
