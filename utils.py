#!/usr/bin/env python3
"""
DataSet class for training UNet
Author: Shilpaj Bhalerao
Date: Sep 19, 2023
"""
# Standard Library Imports
from typing import NoReturn

# Third-Party Imports
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms

def tensor_trimap(t):
    """
    Create a tensor for a segmentation trimap.
    Input: Float tensor with values in [0.0 .. 1.0]
    Output: Long tensor with values in {0, 1, 2}
    """
    x = t * 255
    x = x.to(torch.long)
    x = x - 1
    return x


def args_to_dict(**kwargs):
    """
    Input arguments and return dictionary
    """
    return kwargs


def display_loss_and_accuracies(train_losses: list,
                                test_losses: list,
                                plot_size: tuple = (10, 5)) -> NoReturn:
    """
    Function to display training and test information(losses and accuracies)
    :param train_losses: List containing training loss of each epoch
    :param test_losses: List containing test loss of each epoch
    :param plot_size: Size of the plot
    """
    # Create a plot of 2x2 of size
    fig, axs = plt.subplots(1, 2, figsize=plot_size)

    # Plot the training loss and accuracy for each epoch
    axs[0].plot(train_losses)
    axs[0].set_title("Training Loss")
    axs[1].plot(test_losses)
    axs[1].set_title("Test Loss")


def display_output(model, test_loader):
    """
    Function to predict the output of UNet
    :param model: Trained Model
    :param test_loader:  Test loader
    """
    # Set model to eval model and put in on CPU
    model.eval()
    model.to('cpu')

    # Transform to convert tensor to Image
    transform = transforms.ToPILImage()
    input_img, target_mask = next(iter(test_loader))
    fig, axs = plt.subplots(5, 3, figsize=(15, 25))

    for index in range(5):
        img = input_img[index].unsqueeze(0)
        output = model(img)
        output = nn.Softmax(dim=1)(output)
        predicted_mask = output.argmax(dim=1)
        predicted_mask = predicted_mask.unsqueeze(1).to(torch.float)

        axs[index, 0].imshow(transform(input_img[index]))
        axs[index, 0].set_title('Input Image')
        axs[index, 1].imshow(transform(target_mask[index].float()))
        axs[index, 1].set_title('Target Mask')
        axs[index, 2].imshow(predicted_mask.squeeze(0).squeeze(0))
        axs[index, 2].set_title('Predicted Mask')
