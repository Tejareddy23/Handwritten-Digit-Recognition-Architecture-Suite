"""
Dataset preparation, data augmentation, and PyTorch DataLoaders.

DL Algorithms / Concepts:
- Data Augmentation: Introduces synthetic variability (random rotation, slight affine shifts)
  to regularize the network and prevent memorization/overfitting.
- Feature Normalization: Centers pixel intensities to zero-mean and unit-variance
  (mu = 0.1307, sigma = 0.3081), ensuring stable gradient flow during backpropagation.
- Mini-batching: Enables Stochastic Gradient Descent (SGD / Adam) by loading data in balanced mini-batches.
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from src.config import DATA_DIR, BATCH_SIZE, RANDOM_SEED


def get_transforms(augment=True):
    """
    Build transforms pipeline for training and evaluation.
    """
    mnist_mean = (0.1307,)
    mnist_std = (0.3081,)

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mnist_mean, mnist_std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mnist_mean, mnist_std)
        ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mnist_mean, mnist_std)
    ])

    return train_transform, eval_transform


def get_dataloaders(batch_size=BATCH_SIZE, val_split=0.15, augment=True, num_workers=0):
    """
    Loads MNIST dataset and creates Train, Validation, and Test DataLoaders.
    
    Args:
        batch_size (int): Mini-batch size.
        val_split (float): Fraction of training data reserved for validation.
        augment (bool): Whether to apply data augmentation to training data.
        num_workers (int): Number of subprocesses for data loading (0 recommended on Windows).
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_transform, eval_transform = get_transforms(augment=augment)

    # Full training dataset
    full_train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )

    # Test dataset
    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=eval_transform
    )

    # Split train into train and validation sets
    total_train = len(full_train_dataset)
    val_size = int(total_train * val_split)
    train_size = total_train - val_size

    # Set deterministic generator for reproducibility
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator
    )

    # Note: val_dataset needs eval_transform (no random rotation during validation)
    # We apply a wrapper class to apply eval_transform cleanly
    class TransformedSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __getitem__(self, idx):
            x, y = self.subset.dataset[self.subset.indices[idx]]
            # x is already transformed by the underlying dataset if using subset directly,
            # but for standard torchvision MNIST dataset, raw PIL image can be accessed:
            raw_img = self.subset.dataset.data[self.subset.indices[idx]].numpy()
            from PIL import Image
            img = Image.fromarray(raw_img, mode='L')
            return self.transform(img), y

        def __len__(self):
            return len(self.subset)

    val_dataset = TransformedSubset(val_dataset, eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader

