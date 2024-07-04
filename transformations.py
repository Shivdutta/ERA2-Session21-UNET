#!/usr/bin/env python3
"""
Script containing transformations to be applied on the dataset
Author: Shilpaj Bhalerao
Date: Sep 18, 2023
"""
from torchvision import transforms


TRANSFORMS = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.Normalize((), ()),
        transforms.ToTensor(),
    ]
)


# MASK_TRANSFORMS = transforms.Compose(
#     [
#         transforms.Resize((128, 128)),
#         transforms.Normalize((), ()),
#         transforms.ToTensor(),
#     ]
# )
