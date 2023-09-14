import torch
import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self, scale_factor=2, num_channels=1, n1=64, n2=32, f1=9, f3=5):
        super(SRCNN, self).__init__()
        self.upsample = F.interpolate(scale_factor=scale_factor, mode="bicubic") # 1) might take this out later
        self.patch_mapping = nn.Conv2d(num_channels, n1, kernel_size=f1, padding=f1//2)
        self.non_linear_mapping = nn.Conv2d(n1, n2, kernel_size=f3,padding=f3//2)
        self.reconstruction = nn.Conv2d(n2, num_channels, kernel_size=f3, padding=f3//2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, y):
        y = self.upsample(y) # 1) might take this out later
        f1_y = self.relu(self.patch_mapping(y))
        f2_y = self.relu(self.non_linear_mapping(f1_y))
        f_y = self.reconstruction(f2_y)
        return f_y