import numpy as np
import torch
from tests import _PATH_DATA
from src.models.model import ConvNet
import pytest

n = 16

train_images = torch.unsqueeze(
    torch.load(f"{_PATH_DATA}/processed/train_images.pt")[0:n], dim=1
)
model = ConvNet()
output = model(train_images)


def test_output():
    assert (
        output.shape[1] == 10
    ), "Output shape is expected to match 10, the number of classes"


def test_error_on_wrong_shape():
    with pytest.raises(ValueError, match="Expected input to a 4D tensor"):
        model(torch.randn(1, 2, 3))
