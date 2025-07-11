import torch
import numpy as np
from torch import nn

class SimpleDeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(SimpleDeConvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x



class Generator(nn.Module):
    def __init__(self, input_dim=512, output_channels=3, output_image_dim=128):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_channels = output_channels
        self.output_image_dim = output_image_dim
        
        self.fc = nn.Linear(input_dim, 16*4*4*4)
        self.deconv_blocks = nn.Sequential(
            SimpleDeConvBlock(64, 32, kernel_size=4, stride=2, padding=1),                    #(batch_size, 64, 4, 4)  ----> (batch_size, 32, 8, 8)
            SimpleDeConvBlock(32, 16, kernel_size=4, stride=2, padding=1),                     #(batch_size, 32, 8, 8) -----> (batch_size, 16,16, 16)
            SimpleDeConvBlock(16, 8, kernel_size=4, stride=2, padding=1),                      #(batch_size, 16, 16, 16) ---> (batch_size, 8, 32, 32)
            SimpleDeConvBlock(8, 4, kernel_size=4, stride=2, padding=1),                      #(batch_size, 8, 32, 32) ---> (batch_size, 4, 64, 64)
            nn.ConvTranspose2d(4, 3, kernel_size=4, stride=2, padding=1),                      #(batch_size, 4, 64, 64) --> (batch_size, 3, 128, 128)
            nn.Tanh()  # Output activation
        )
        
    def forward(self, x):  # input shape: (batch_size, 512)       
        x = x+ 0.01*torch.randn_like(x)  # Adding noise to the input
        x = self.fc(x)
        x = x.view(-1,16*4,4,4)                        # Reshape to (batch_size, 256, 4, 4)
        x = self.deconv_blocks(x)                      # output shape: (batch_size, 3, 128, 128)
        return x


class SimpleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(SimpleConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.pool(x)
        return x

#using a simple fastvit image encoder for discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv_head= nn.Sequential(
            SimpleConvBlock(3, 4, kernel_size=4, stride=2, padding=1),   #(batch_size, 3, 128, 128) --> (batch_size, 4, 32, 32)
            SimpleConvBlock(4, 8, kernel_size=4, stride=2, padding=1),  #(batch_size, 4, 32, 32) ----> (batch_size, 8, 8, 8)
            SimpleConvBlock(8, 16, kernel_size=4, stride=2, padding=1), #(batch_size, 8, 8, 8) -----> (batch_size, 16, 2, 2)
        )
        
        self.linear_head = nn.Sequential(
            nn.Flatten(),  # Flatten the output from conv layers
            nn.Linear(16 * 2 * 2, 16),  # Fully connected layer to reduce dimensions
            nn.GELU(),
            nn.Linear(16, 1)  # Final output layer for binary classification
        )

    def forward(self, x):
        return self.linear_head(self.conv_head(x))  # Output: [batch_size,3,128,128]--> [batch_size, 1]

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
    generator = Generator()
    disc = Discriminator()

    random_tensor = torch.randn(32, 512,4,4)

    output = generator(random_tensor)
    print("Generator Output Shape:", output.shape)  
    
    # disc_output = disc(output)
    # print("Discriminator Output Shape:", disc_output.shape) 
