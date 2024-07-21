import torch

from utils import save_checkpoint, load_checkpoint
import config

import torch.nn as nn
import torch.optim as optim



#### Switch to using the CTGTDataset class from the dataset module if there is only one location of the input and ground truth images respectively ####
#from dataset import CTGTDataset
from dataset_multi import CTGTDataset

from generator_model import Generator
from discriminator_model import Discriminator

from torch.utils.data import DataLoader
from tqdm import tqdm

import os

import csv


def train_fn(epoch, disc, gen, loader, opt_disc, opt_gen, l2, bce, g_scaler, d_scaler):
        # Open a file to write the losses
     with open(f'training_losses.csv', mode='w', newline='') as file:
         writer = csv.writer(file)
         # Write the header
         writer.writerow(["Epoch", "Batch", "D_Loss", "G_Loss"])
         loop = tqdm(loader, leave=True)

         for idx, (x, y) in enumerate(loop):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)

            # Train Discriminator
            with torch.cuda.amp.autocast():
                y_fake = gen(x)
                D_real = disc(x, y)
                D_fake = disc(x, y_fake.detach())
                D_real_loss = bce(D_real, torch.ones_like(D_real))
                D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2

            disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

            # Train generator

            with torch.cuda.amp.autocast():
                D_fake = disc(x, y_fake)
                G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
                L2 = l2(y_fake, y) * config.L1_LAMBDA
                G_loss = G_fake_loss + L2

            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()
            
            writer.writerow([epoch, idx, D_loss.item(), G_loss.item()])

            loop.set_description(f"D Loss: {D_loss.item():.4f}, G Loss: {G_loss.item():.4f}")

def main():
    print("using device: ", config.DEVICE)
    print("batch size: ", config.BATCH_SIZE)
    print("num epochs: ", config.NUM_EPOCHS)

    disc = Discriminator(in_channels=2).to(config.DEVICE)
    gen = Generator(in_channels=1, out_channels=1).to(config.DEVICE)
    
    

    optim_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    optim_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    BCE = nn.BCEWithLogitsLoss()
    L2_LOSS = nn.MSELoss()





    # input_dir = "PATH_TO_INPUT_IMAGES"  # Uncomment if using single directory CTGTDataset 
    # output_dir = "PATH_TO_TARGET_IMAGES" # Uncomment if using single directory CTGTDataset 
    #train_dataset = CTGTDataset(input_dir=input_dir, # Uncomment if using single directory CTGTDataset 
    #                            target_dir=output_dir) # Uncomment if using single directory CTGTDataset 
    
    
    input_names = ["input1", "input2", "input3", "input4"] # Comment out if using single directory CTGTDataset 
    target_names = ["target1", "target2", "target3", "target4"]  # Comment out if using single directory CTGTDataset 
    train_dataset = CTGTDataset(input_names,  # Comment out if using single directory CTGTDataset 
                                target_names)  # Comment out if using single directory CTGTDataset 
    
    



    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    gen_scaler = torch.cuda.amp.GradScaler()
    disc_scaler = torch.cuda.amp.GradScaler()
    
    # load_checkpoint(f"path_to_generator_checkpoint", gen, optim_gen, "cuda") # Uncomment to load a checkpoint
    # load_checkpoint(f"path_to_discriminator_checkpoint", disc, optim_disc, "cuda") # Uncomment to load a checkpoint

    CHECKPOINT_FREQUENCY = 10 # Interval of number of epochs to save a checkpoint. NOTE: Will always save the final model.

    for epoch in range(0, config.NUM_EPOCHS):
        print(f"epoch: {epoch}")
        train_fn(epoch, disc, gen, train_loader, optim_disc, optim_gen, L2_LOSS, BCE, gen_scaler, disc_scaler)

        if ((epoch % CHECKPOINT_FREQUENCY == 0) or (epoch == (config.NUM_EPOCHS - 1))):
            save_checkpoint(gen, optim_gen, filename=f"generator{epoch}.pth.tar")
            save_checkpoint(disc, optim_disc, filename=f"discriminator{epoch}.pth.tar")
        

if __name__ == "__main__":
    main()