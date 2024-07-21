from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms



class CTGTDataset(Dataset):
    """
    This version of the CTGTDataset allows you to use multiple directory locations for the training dataset.

    CTGT stands for CrossTalk GroundTruth. 

    The input dirs should be a list of paths to images of the channel with crosstalk of a second channel.
    The output dirs should be a list of paths to images of the ground truth signal of the second channel.

    Input directories and output directories are coupled by their index. 

    Files between coupled directories are coupled by their name. Ensure the same name for images that go together.
    (For example: you could name the files frame_00001.png, frame_00002.png etc in both directories)
    """
    def __init__(self, input_dirs, target_dirs, transform=None):
        self.input_dirs = input_dirs
        self.target_dirs = target_dirs
        
        # Initialize lists to store files
        self.list_input = []
        self.list_target = []

        # Iterate over directories and gather common files
        for input_dir, target_dir in zip(self.input_dirs, self.target_dirs):
            input_files = sorted(os.listdir(input_dir))
            target_files = sorted(os.listdir(target_dir))

            set_input = set(input_files)
            set_target = set(target_files)

            common_files = set_input & set_target

            self.list_input.extend([os.path.join(input_dir, filename) for filename in common_files])
            self.list_target.extend([os.path.join(target_dir, filename) for filename in common_files])

        # Ensure the lengths match after filtering
        if len(self.list_input) != len(self.list_target):
            raise ValueError(f"The number of input and target files do not match. Lengths {len(self.list_input)} and {len(self.list_target)}")

        # Define the transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((240, 300)),  # Downscale the image
                transforms.ToTensor()  # Should also linearly scale between zero and one
            ])

    def __len__(self):
        return len(self.list_input)

    def __getitem__(self, index):
        input_path = self.list_input[index]
        target_path = self.list_target[index]

        input_image = self.transform(Image.open(input_path).convert('L'))
        target_image = self.transform(Image.open(target_path).convert('L'))

        return input_image, target_image
