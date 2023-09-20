import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from utils import load_recording, normalize_ecg
from scipy import signal
import numpy as np
import random


def create_dataloader(opt):
    """Create training and validation data loaders."""
    print('Load the dataset...')

    # Read train and validation datasets from CSV
    train_header = pd.read_csv('./dataset/train_dataset.csv')
    valid_header = pd.read_csv('./dataset/test_dataset.csv')
    train = pd.DataFrame(train_header)
    valid = pd.DataFrame(valid_header)

    print(len(train))

    # Create training data loader
    train_loader = DataLoader(
        dataset=ECG_Dataset(opt, train, mode='train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        sampler=None
    )

    # Create validation data loader
    validatioin_loader = DataLoader(
        dataset=ECG_Dataset(opt, valid, mode='valid'),
        batch_size=opt.batch_size, shuffle=False, num_workers=0
    )

    return train_loader, validatioin_loader


def create_dataloader_for_test(opt):
    """Create test data loader."""

    print('Load the dataset...')
    if not os.path.exists('./dataset/test_dataset.csv'):
        exit()  # Exit if test dataset CSV doesn't exist
    else:
        test = pd.read_csv('./dataset/test_dataset.csv')
        test = pd.DataFrame(test)
        print(len(test))

    # Create test data loader
    test_loader = DataLoader(
        dataset=ECG_Dataset(opt, test, mode='test'),
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    return test_loader


class ECG_Dataset(Dataset):
    """ECG Dataset class that implements the required methods for PyTorch's Dataset class."""
    def __init__(self, opt, dataset, mode):
        # Initialize variables and load data
        self.fs = opt.fs
        self.samples = opt.samples
        self.mode = mode
        self.dataset = dataset

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Return an item from the dataset at the specified index."""
        fs = self.dataset.iloc[idx]['fs']  # Fetch the sampling frequency
        targets = self.dataset.iloc[idx]['target']  # Fetch the targets

        # Load ECG recording
        inputs = load_recording(self.dataset.iloc[idx]['record'])

        # [Step 1] Normalization: Use z-score normalization
        inputs = normalize_ecg(inputs)

        # [Step 2] Resample data to 500Hz if needed
        if fs == float(1000):
            inputs = signal.resample_poly(inputs, up=1, down=2, axis=-1)  # to 500Hz
        elif fs == float(500):
            pass
        else:
            inputs = signal.resample(inputs, int(inputs.shape[1] * 500 / fs), axis=1)

        inputs = np.nan_to_num(inputs)  # Convert NaN values to zero
        inputs = torch.from_numpy(inputs)  # Convert to PyTorch tensor

        # [Step 3] Zero-Padding: Adjust input length to a fixed size (self.samples)
        if inputs.size(1) > self.samples:
            start_index = random.randint(0, inputs.size(1) - self.samples - 1)
            inputs = inputs[:, start_index:start_index + self.samples]
        else:
            pad_len = self.samples - inputs.size(1)
            inputs = torch.cat([inputs, torch.zeros([inputs.size(0), pad_len])], dim=1)

        # Process and convert targets to tensor
        targets = list(map(float, targets.replace('[', '').replace(']', '').split()))
        targets = torch.Tensor(targets)

        return inputs, targets
#