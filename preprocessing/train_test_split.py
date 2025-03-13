import os
import shutil
import random

### VARIABLES TO SET ###
NUM_FILES = 3857 # How many files?
TEST_PERC = 20 # What percentage of them should be in the test set?

# Define the source directory containing the subfolders
source_dir = "/media/nick/C8EB-647B/Data/processed/2_chs/augmented/Exp_1"

#######################

# Generate a list of VAL_PERC% of NUM_FILES indices to put in the TEST set
random_indices = random.sample(range(NUM_FILES), int(NUM_FILES * (TEST_PERC/100)))

# Iterate through each subdirectory in the source directory
for subdir in os.listdir(source_dir):
    print(subdir)
    full_subdir_path = os.path.join(source_dir, subdir)

    # Check if it's a directory
    if os.path.isdir(full_subdir_path):
        # Create train and val subdirectories
        train_subdir = os.path.join(full_subdir_path, 'train')
        val_subdir = os.path.join(full_subdir_path, 'test')
        os.makedirs(train_subdir, exist_ok=True)
        os.makedirs(val_subdir, exist_ok=True)

        # Move files based on the random indices
        for i in range(NUM_FILES):
            filename = f"frame_{i:05d}"
            source_file = os.path.join(full_subdir_path, filename)
            
            # Check if the file exists to prevent errors
            if os.path.exists(source_file):
                if i in random_indices:
                    shutil.move(source_file, os.path.join(val_subdir, filename + ".png"))
                else:
                    shutil.move(source_file, os.path.join(train_subdir, filename + ".png"))

print("Files have been organized into train and val directories.")
