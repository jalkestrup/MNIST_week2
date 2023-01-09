import numpy as np
import torch
from tests import _PATH_DATA
from src.models.model import ConvNet

n = 16
train_images = torch.unsqueeze(torch.load(f"{_PATH_DATA}/processed/train_images.pt")[0:n], dim=1)
model = ConvNet()
output = model(train_images)

def test_output():
    assert output.shape[1] == 10

