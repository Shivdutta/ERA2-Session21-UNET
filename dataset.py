#!/usr/bin/env python3
"""
DataSet class for training UNet
Author: Shilpaj Bhalerao
Date: Sep 18, 2023
"""
# Standard Library Imports
import os

# Third-Party Imports
import torch
from torchvision import datasets


class OxfordIIITPetsAugmented(datasets.OxfordIIITPet):
    """
    Create a dataset wrapper that allows us to perform custom image augmentations on both the target and label (segmentation mask) images.
    """
    def __init__(
            self,
            root: str,
            split: str,
            target_types="segmentation",
            download=False,
            pre_transform=None,
            post_transform=None,
            pre_target_transform=None,
            post_target_transform=None,
            common_transform=None,
    ):
        super().__init__(
            root=root,
            split=split,
            target_types=target_types,
            download=download,
            transform=pre_transform,
            target_transform=pre_target_transform,
        )
        self.post_transform = post_transform
        self.post_target_transform = post_target_transform
        self.common_transform = common_transform

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        """
        Common transforms are performed on both the input and the labels
        by creating a 4 channel image and running the transform on both.
        Then the segmentation mask (4th channel) is separated out.
        """
        _input, target = super().__getitem__(idx)

        if self.common_transform is not None:
            both = torch.cat([_input, target], dim=0)
            both = self.common_transform(both)
            (_input, target) = torch.split(both, 3, dim=0)

        if self.post_transform is not None:
            _input = self.post_transform(_input)
        if self.post_target_transform is not None:
            target = self.post_target_transform(target)

        return _input, target
