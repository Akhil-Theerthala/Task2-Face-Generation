import torch
import numpy as np
from torch import nn
from torch.nn.utils import spectral_norm

class SimpleDeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(SimpleDeConvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)


    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x

class UpBlock(nn.Module):
    """
    Nearest neighbour ↑2 followed by a 3×3 stride-1 conv.
    PixelNorm (or BatchNorm) + LeakyReLU afterwards.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),           #  H, W  → 2H, 2W
            nn.Conv2d(in_c, out_c, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x): return self.net(x)


class Generator(nn.Module):
    def __init__(self, input_dim=512, output_channels=3, output_image_dim=128):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_channels = output_channels
        self.output_image_dim = output_image_dim
        self.image_dim = 512
        self.fc = nn.Linear(input_dim, self.image_dim*4*4)
        self.deconv_blocks = nn.Sequential(
            SimpleDeConvBlock(self.image_dim,  self.image_dim//2),                   
            SimpleDeConvBlock(self.image_dim//2,  self.image_dim//4),                 
            SimpleDeConvBlock(self.image_dim//4,  self.image_dim//8),                 
            UpBlock(self.image_dim//8, self.image_dim//16),                   
            UpBlock(self.image_dim//16, self.image_dim//32),                             
            # UpBlock(128,  64),                           
            # nn.Upsample(scale_factor=1, mode='nearest'),     
            nn.Conv2d(self.image_dim//32, output_channels, 3, 1, 1),  
            nn.Tanh()
        )
        
    def forward(self, x):  # input shape: (batch_size, 512)       
        x = x+ 0.1*torch.randn_like(x)  # Adding noise to the input
        x = self.fc(x)
        x = x.view(-1,self.image_dim,4,4)                        # Reshape to (batch_size, 512, 4, 4)
        x = self.deconv_blocks(x)                      # output shape: (batch_size, 3, 64, 64)
        return x


class SimpleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(SimpleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x

#using a simple fastvit image encoder for discriminator
class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        self.start_dim = 32
        
        self.conv_head= nn.Sequential(
            SimpleConvBlock(input_channels, self.start_dim, kernel_size=4, stride=2, padding=1),    #(batch_size, 3, 32, 32) --> (batch_size, 32, 16, 16)
            SimpleConvBlock(self.start_dim, self.start_dim*2, kernel_size=4, stride=2, padding=1),  #(batch_size, 32, 16, 16) ----> (batch_size, 64, 8, 8)
            SimpleConvBlock(self.start_dim*2, self.start_dim*4, kernel_size=4, stride=2, padding=1), #(batch_size,  64, 8, 8) -----> (batch_size, 128, 4, 4)
            SimpleConvBlock(self.start_dim*4, self.start_dim*4, kernel_size=4, stride=2, padding=1), #(batch_size, 128, 4, 4) -----> (batch_size, 256, 2, 2)
            SimpleConvBlock(self.start_dim*4, self.start_dim*8, kernel_size=4, stride=2, padding=1), #(batch_size, 128, 4, 4) -----> (batch_size, 256, 2, 2)
            SimpleConvBlock(self.start_dim*8, self.start_dim*16, kernel_size=4, stride=2, padding=1), #(batch_size, 128, 4, 4) -----> (batch_size, 256, 2, 2)
            nn.Conv2d(self.start_dim*16, 1, kernel_size=2, stride=1, padding=0)  # Final output layer for binary classification
        )


    def forward(self, x):
        x = self.conv_head(x)  # Input shape: [batch_size, 3, 128, 128] Output shape: [batch_size, 1, 1, 1]
        x = x.view(x.size(0), 1)  # Flatten to [batch_size, 1]
        return x

def weights_init_normal(m):
    """
    Applies initial weights to certain layers in a model .
    The weights are taken from a normal distribution 
    with mean = 0, std dev = 0.02.
    :param m: A module or layer in a network    
    """
    # classname will be something like:
    # `Conv`, `BatchNorm2d`, `Linear`, etc.
    classname = m.__class__.__name__
    
    # TODO: Apply initial weights to convolutional and linear layers
    if classname.find('Linear') != -1:
        n = m.in_features
        y = (1.0/np.sqrt(n))
        m.weight.data.normal_(0, y)
        m.bias.data.fill_(0)
    
    if classname.find('Conv2d') != -1:
        n = m.in_channels
        y = (1.0/np.sqrt(n))
        m.weight.data.normal_(0, y)

    if classname.find('ConvTranspose2d') != -1:
        n = m.in_channels
        y = (1.0/np.sqrt(n))
        m.weight.data.normal_(0, y)
    


if __name__ == "__main__":
    generator = Generator(output_channels=1)
    disc = Discriminator(input_channels=1)

    random_tensor = torch.randn(32,512)

        
    output =generator(random_tensor)
    print("Generator Output Shape:", output.shape)  
    
    
    disc_output = disc(output)
    # print("Discriminator Output Shape:", disc_output.view(disc_output.size(0), -1).shape) 
    print("Raw Discriminator Output Shape:", disc_output.shape) 
