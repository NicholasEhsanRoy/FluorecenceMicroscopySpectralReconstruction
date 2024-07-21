import torch
import torch.nn as nn
import config
from PIL import Image, ImageDraw
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage, ToTensor
import os
import csv

"""

This module consists of a series of utility functions for this project.

"""




def concat_images(input, real, fake):
    """
    Function used to create the Visual Results figures in the associated paper.
    Concatenates the input, ground truth (real), and predicted (fake) images, with 2 pixel wide vertical lines between them
    """
    max_height = max(input.height, real.height, fake.height)
    total_width = input.width + real.width + fake.width + 4  # Add 4 for the white lines

    new_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))

    x_offset = 0
    for image in (input, real, fake):
        new_image.paste(image, (x_offset, 0))
        x_offset += image.width + 2  # Add 2 for the white line
        # Add the white line
        draw = ImageDraw.Draw(new_image)
        draw.line((x_offset, 0, x_offset, max_height), fill=(255, 255, 255), width=2)
        x_offset += 2  # Add 2 for the next image
    return new_image

def concat_two_images(input, output):
    """
    Useful if you only want to couple the input with either the ground truth or the prediction
    """
    max_height = max(input.height, output.height)
    total_width = input.width + output.width + 4
    
    new_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    
    x_offset = 0
    
    for image in (input, output):
        new_image.paste(image, (x_offset, 0))
        x_offset += image.width + 2
        
        draw = ImageDraw.Draw(new_image)
        draw.line((x_offset, 0, x_offset, max_height), fill=(255, 255, 255), width=2)
        x_offset += 2
    return new_image

def save_example(gen, val_loader, epoch, batch, folder):
    L1_LOSS = nn.L1Loss()
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        loss = L1_LOSS(y, y_fake)
        print("L1 loss:", loss.item())

        # Handle the case for batch size of 1 or more
        if x.dim() == 4:  # Check if batched input (B, C, H, W)
            x_demo = x[0].detach().cpu()  # Use the first image in the batch for demonstration
            y_demo = y[0].detach().cpu()
            y_fake_demo = y_fake[0].detach().cpu()
        else:  # Handle non-batched input (C, H, W)
            x_demo = x.detach().cpu()
            y_demo = y.detach().cpu()
            y_fake_demo = y_fake.detach().cpu()

        # Convert tensors to PIL Images
        to_pil = ToPILImage()
        x_pil = to_pil(x_demo)
        y_pil = to_pil(y_demo)
        y_fake_pil = to_pil(y_fake_demo)

        concat = concat_images(x_pil, y_pil, y_fake_pil)
        concat.save(f"{dir}epoch_{epoch}-batch_{batch}.png")
    gen.train()

def save_checkpoint(model, optimizer, filename="checkpoint.pth.tar", additional_info=None):
    """
    Used to save checkpoints of the models when training
    """
    print("=> Saving Checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    if additional_info is not None:
        checkpoint.update(additional_info)
    
    # Try to save the checkpoint
    try:
        torch.save(checkpoint, filename)
        print("=> Checkpoint was successfully saved")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")


def load_checkpoint(filename, model, optimizer, device):
    """
    Used to load a checkpoint for continued training
    """
    print(f"=> Loading checkpoint from '{filename}'")
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if 'additional_info' in checkpoint:
        return checkpoint['additional_info']  # Return additional info if exists

    
def load_checkpoint2(filename, model, optimizer=None, device='cuda'):
    """
    Used to load a checkpoint for inference
    """
    print(f"=> Loading checkpoint from '{filename}'")
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if 'additional_info' in checkpoint:
        return checkpoint['additional_info']  # Return additional info if exists
    
    
def calculate_metrics(prediction, ground_truth, threshold, do_print=False):
    """
    Calculates Precision, Recall, and F1 for the a given prediction, ground truth, and similarity threshold.

    The threshold determines how close two pixels need to be to be considered correctly identified.

    True positives are defined as a correct identification of a singal

    False positives are defined as when the model predicts a signal at a pixel, but there isn't one

    False negatives are defined as when the model doesn't recognize a signal where there is one.
    """

    zero_threshold = 10  # Adjust this value based on the actual range of possible values for the pixels

    # Convert to PIL image and back to Tensor to normalize to 0-255
    to_pil = ToPILImage()
    to_tensor = ToTensor()
    
    prediction = to_tensor(to_pil(prediction[0].cpu())) * 255
    ground_truth = to_tensor(to_pil(ground_truth[0].cpu())) * 255
    


    prediction = prediction.to(ground_truth.device)  # Ensure both tensors are on the same device

    abs_diff = torch.abs(prediction - ground_truth)
    correct = abs_diff < threshold

    # correctly identified signal
    true_positives = torch.sum(correct & (ground_truth >= zero_threshold))
    
    # Thinks there is signal here but there isn't
    false_positives = torch.sum(~correct & (ground_truth < zero_threshold))
    
    # Doesn't think there is signal here, but there is
    false_negatives = torch.sum(~correct & (ground_truth >= zero_threshold))

    precision = true_positives / (true_positives + false_positives + 1e-9)
    recall = true_positives / (true_positives + false_negatives + 1e-9)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)
    
    if do_print:
        print("min prediction: ", prediction.min().item())
        print("max prediction: ", prediction.max().item())

    return precision.item(), recall.item(), f1_score.item()
    
