'''
model.py

GunGyeom James Kim
September 28th, 2023
CS 7180: Advnaced Perception

architecture of SRCNN
'''
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self, scale_factor=3, num_channels=1, n1=64, n2=32, f1=9, f3=5):
        '''
        constructor

        Parameters:
            scale_factor - factor for scaling, e.g. 2, 3 and 4
            num_channels - number of channels in the input
            n1 - output size of patch mapping layer
            n2 - output size of non-linear mapping layer
            f1 - size of kernel of patch mapping layer
            f3 - size of kernel of non-linear mapping layer
        '''
        super(SRCNN, self).__init__()
        self.patch_mapping = nn.Conv2d(num_channels, n1, kernel_size=f1, padding=f1//2)
        self.non_linear_mapping = nn.Conv2d(n1, n2, kernel_size=f3,padding=f3//2)
        self.reconstruction = nn.Conv2d(n2, num_channels, kernel_size=f3, padding=f3//2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, y):
        '''
        Return the output of the network

        Parameters:
            y: input of the network

        Return:
            f_y: output of the network
        '''
        f1_y = self.relu(self.patch_mapping(y))
        f2_y = self.relu(self.non_linear_mapping(f1_y))
        f_y = self.reconstruction(f2_y)
        return f_y