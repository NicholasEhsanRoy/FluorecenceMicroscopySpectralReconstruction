from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms

class CTGTDataset(Dataset):
    """
    The standard version of the CTGTDataset.

    CTGT stands for CrossTalk GroundTruth. 

    The input dirs should be a path to the directory of input images - images of a color channel with crosstalk of a second channel.
    The output dirs should be the path to images of the ground truth signal of the second channel.

    Files between input and output directories are coupled by their name. Ensure the same name for images that go together.

    (For example: you could name the files frame_00001.png, frame_00002.png etc in both directories)
    """
    def __init__(self, input_dir, target_dir, transform = None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        
        self.list_input = sorted(os.listdir(self.input_dir))
        self.list_target = sorted(os.listdir(self.target_dir))

        if len(self.list_input) != len(self.list_target):
            set_input = set(self.list_input)
            set_target = set(self.list_target)
            
            common_files = set_input & set_target
            
            # Filter the original lists to include only the common files
            self.list_input = [filename for filename in self.list_input if filename in common_files]
            self.list_target = [filename for filename in self.list_target if filename in common_files]

            # Check if the lengths are the same and if not, raise an error
            if len(self.list_input) != len(self.list_target):
                raise ValueError(f"After filtering, the number of input and target files do not match. Lengths {len(self.list_input)} and {len(self.list_target)}")

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((240, 300)), # Downscale the image
                transforms.ToTensor() # Should also linearly scale between zero and one
            ])

    def __len__(self):
        return len(self.list_input)
    
    def __getitem__(self, index):
        input_file = self.list_input[index]
        target_file = self.list_target[index]

        input_path = os.path.join(self.input_dir, input_file)
        target_path = os.path.join(self.target_dir, target_file)

        input = self.transform(Image.open(input_path).convert('L'))
        target = self.transform(Image.open(target_path).convert('L'))

        return input, target