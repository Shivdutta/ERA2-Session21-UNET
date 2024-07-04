#!/usr/bin/env python3
"""
Utilities for Model Training
Author: Shilpaj Bhalerao
Date: Jun 21, 2023
"""
# Standard Library Imports

# Third-Party Imports
import torch
import torch.nn.functional as F
from tqdm import tqdm


def get_correct_predictions(prediction, labels):
    """
    Function to return total number of correct predictions
    :param prediction: Model predictions on a given sample of data
    :param labels: Correct labels of a given sample of data
    :return: Number of correct predictions
    """
    return prediction.argmax(dim=1).eq(labels).sum().item()


def train(model, device, train_loader, optimizer, criterion, scheduler=None):
    """
    Function to train model on the training dataset
    :param model: Model architecture
    :param device: Device on which training is to be done (GPU/CPU)
    :param train_loader: DataLoader for training dataset
    :param optimizer: Optimization algorithm to be used for updating weights
    :param criterion: Loss function for training
    :param scheduler: Scheduler for learning rate
    """
    # Enable layers like Dropout for model training
    model.train()

    # Utility to display training progress
    pbar = tqdm(train_loader)

    # Variables to track loss and accuracy during training
    train_loss = 0

    # Iterate over each batch and fetch images and labels from the batch
    for batch_idx, (data, target) in enumerate(pbar):

        # Put the images and labels on the selected device
        data, target = data.to(device), target.to(device)

        # Reset the gradients for each batch
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target.squeeze(1))
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Use learning rate scheduler if defined
        if scheduler:
            scheduler.step()

        # Display the training information
        pbar.set_description(
            desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx}')

    return train_loss


def test(model, device, test_loader, criterion):
    """
    Function to test the model training progress on the test dataset
    :param model: Model architecture
    :param device: Device on which training is to be done (GPU/CPU)
    :param test_loader: DataLoader for test dataset
    :param criterion: Loss function for test dataset
    """
    # Disable layers like Dropout for model inference
    model.eval()

    # Variables to track loss and accuracy
    test_loss = 0

    # Disable gradient updation
    with torch.no_grad():
        # Iterate over each batch and fetch images and labels from the batch
        for batch_idx, (data, target) in enumerate(test_loader):

            # Put the images and labels on the selected device
            data, target = data.to(device), target.to(device)

            # Pass the images to the output and get the model predictions
            output = model(data)
            test_loss += criterion(output, target.squeeze(1)).item()  # sum up batch loss

    # Calculate test loss for a epoch
    test_loss /= len(test_loader)

    print('Test set: Average loss: {:.4f}'.format(test_loss))

    return test_loss


def dice_loss(logits, true, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Reference Link: https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py

    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.

    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.

    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes).to(device)[true]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return 1 - dice_loss
