import torch
import torch.nn as nn

"""
Implementation of a Convolutional Neural Network (CNN) PatchGAN discriminator.
Inspired by the original Pix2Pix paper and this youtube video: https://www.youtube.com/watch?v=SuddDSqGRzg
"""

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, bias=False, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)
    

# PatchGAN CNN discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=2, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature
        
        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            )
        )
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, y):
        x = torch.cat([x, y], dim = 1) # concatenate the images accross the channels
        x = self.initial(x)
        return self.model(x)


def test():
    """
    The sizes of the images during training
    """
    x = torch.randn([1, 1, 240, 300])
    y = torch.randn([1, 1, 240, 300])
    model = Discriminator()
    preds = model(x, y)
    print(preds.shape)




if __name__ == "__main__":
    test()