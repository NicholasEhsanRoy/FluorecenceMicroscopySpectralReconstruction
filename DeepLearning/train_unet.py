import torch
from utils import save_checkpoint, load_checkpoint
import config

import torch.nn as nn
import torch.optim as optim

#### Switch to using the CTGTDataset class from the dataset module if there is only one location of the input and ground truth images respectively ####
#from dataset import CTGTDataset
from dataset_multi import CTGTDataset

from generator_model import Generator

from torch.utils.data import DataLoader
from tqdm import tqdm

import os

import csv

def train_fn(epoch, gen, loader, opt_gen, mse, scaler):
    with open(f'training_losses.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Batch", "G_Loss"])
        loop = tqdm(loader, leave=True)

        for idx, (x, y) in enumerate(loop):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)

            with torch.cuda.amp.autocast():
                y_pred = gen(x)
                G_loss = mse(y_pred, y)

            opt_gen.zero_grad()
            scaler.scale(G_loss).backward()
            scaler.step(opt_gen)
            scaler.update()

            writer.writerow([epoch, idx, G_loss.item()])
            loop.set_description(f"G Loss: {G_loss.item():.4f}")

def main():
    print("using device: ", config.DEVICE)
    print("batch size: ", config.BATCH_SIZE)
    print("num epochs: ", config.NUM_EPOCHS)
    
    gen = Generator(in_channels=1, out_channels=1, features=16).to(config.DEVICE)
    optim_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    MSE = nn.MSELoss()
    

    # input_dir = "PATH_TO_INPUT_IMAGES"  # Uncomment if using single directory CTGTDataset 
    # output_dir = "PATH_TO_TARGET_IMAGES" # Uncomment if using single directory CTGTDataset 
    #train_dataset = CTGTDataset(input_dir=input_dir, # Uncomment if using single directory CTGTDataset 
    #                            target_dir=output_dir) # Uncomment if using single directory CTGTDataset 
    


    ### Use your own paths to the directories instead of input/target
    input_names = ["input1", "input2", "input3", "input4"] # Comment out if using single directory CTGTDataset 
    target_names = ["target1", "target2", "target3", "target4"]  # Comment out if using single directory CTGTDataset 
    train_dataset = CTGTDataset(input_names,  # Comment out if using single directory CTGTDataset 
                                target_names)  # Comment out if using single directory CTGTDataset 

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    scaler = torch.cuda.amp.GradScaler()
    
    # Uncomment to load a checkpoint
    # load_checkpoint(f"unet_path.pth.tar", gen, optim_gen, "cuda")

    CHECKPOINT_FREQUENCY = 10 # Interval of number of epochs to save a checkpoint. NOTE: Will always save the final model.

    for epoch in range(10, config.NUM_EPOCHS):
        print(f"epoch: {epoch}")
        train_fn(epoch, gen, train_loader, optim_gen, MSE, scaler)

        if ((epoch % CHECKPOINT_FREQUENCY == 0) or (epoch == (config.NUM_EPOCHS - 1))):
            save_checkpoint(gen, optim_gen, filename=f"unet_{epoch}.pth.tar")

if __name__ == "__main__":
    main()
