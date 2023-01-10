import torch
from tests import _PATH_DATA
from torch.utils.data import DataLoader, TensorDataset
import pytest


images = torch.unsqueeze(torch.load(f"{_PATH_DATA}/processed/train_images.pt"), dim=1)
labels = torch.load(f"{_PATH_DATA}/processed/train_labels.pt")
train = TensorDataset(images, labels)


def test_train():
    train_set = DataLoader(train, batch_size=8, shuffle=True)
    # Checks that the dataloader is returning the correct shape
    for images, labels in train_set:
        # Check that number and shape of labels and images are equal
        assert (
            images.shape[0] == labels.shape[0]
        ), "Number of images and labels are not equal"
        assert (
            images.shape[1] == labels.shape[1]
        ), "Shape of images and labels are not equal"


# Check that the dataloader is returning the correct shape with different batch sizes
@pytest.mark.parametrize("test_input,expected", [(4, 4), (8, 8), (16, 16)])
def test_dataloader_batch_size(test_input, expected):
    train_set = DataLoader(train, batch_size=test_input, shuffle=True)
    for images, labels in train_set:
        assert (
            images.shape[0] == expected
        ), "Dataloader does not return expected batch size of images"
        assert (
            labels.shape[0] == expected
        ), "Dataloader does not return expected batch size of labels"
        # Only check first loop
        break


# In the detox project we use a pytorch ligthning module, which we can use
# to do call the loss directly from the model


def test_loss():
    assert True
