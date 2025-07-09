import torch
import timm
import numpy as np
from torch import nn

class MappingNetwork(nn.Module):
    def __init__(
        self,
        input_dim = 512,
        output_dim=128):
    
        super(MappingNetwork, self).__init__()
        self.mapping = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.GELU(),
            nn.Linear(input_dim//2, input_dim//4),
            nn.GELU(),
            nn.Linear(input_dim//4, output_dim),
        )
        self.output_dim = output_dim
    
    def forward(self, x):
        # I orirginally thought to add some noise here, but since the original embeddings are anyway being transformed by the network, I am excluding it for now. 
        return self.mapping(x)

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
    def __init__(self, input_dim=128, output_channels=3, output_image_dim=128):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_channels = output_channels
        self.output_image_dim = output_image_dim
        
        self.mapping_network = MappingNetwork()
        
        self.deconv_blocks = nn.Sequential(
            SimpleDeConvBlock(128, 128, kernel_size=4, stride=1, padding=0),                    #(batch_size, 128, 1, 1) --> (batch_size, 128, 4, 4)
            SimpleDeConvBlock(128, 64, kernel_size=4, stride=2, padding=1),                     #(batch_size, 128, 4, 4) --> (batch_size, 64, 8, 8)
            SimpleDeConvBlock(64, 32, kernel_size=4, stride=2, padding=1),                      #(batch_size, 64, 8, 8) --> (batch_size, 32, 16, 16)
            SimpleDeConvBlock(32, 16, kernel_size=4, stride=2, padding=1),                      #(batch_size, 32, 16, 16) --> (batch_size, 16, 32, 32)
            SimpleDeConvBlock(16, 8, kernel_size=4, stride=2, padding=1),                       #(batch_size, 16, 32, 32) --> (batch_size, 8, 64, 64)
            nn.ConvTranspose2d(8, output_channels, kernel_size=4, stride=2, padding=1),         #(batch_size, 8, 64, 64) --> (batch_size, 3, 128, 128)
            nn.Tanh()  # Output activation
        )
        
    def forward(self, x):
        x = self.mapping_network(x)                     #output shape: (batch_size, 128)
        x = x.view(x.size(0), self.input_dim, 1, 1)     #output shape: (batch_size, 128, 1, 1)
        x = self.deconv_blocks(x)                       #output shape: (batch_size, 3, 128, 128)
        return x


#using a simple fastvit image encoder for discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = timm.create_model('fastvit_sa12.apple_in1k', pretrained=False, num_classes=0)
        self.data_config = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.transform = timm.data.create_transform(**self.data_config, is_training=True)
        
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(self.model(x))  # Output: [batch_size,3,128,128]--> [batch_size, 1024] --> [batch_size, 1]


# Since the original DCGAN paper mentioned initializing weights for faster convergence, I am using the same initialization here.
def weights_init_normal(m):     
    classname = m.__class__.__name__
    
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
    
    #apply weights initialization from the DCGAN paper
    generator= generator.apply(weights_init_normal)
    disc = disc.apply(weights_init_normal)
    
    
    random_tensor = torch.randn(32, 512)
    
    output = generator(random_tensor)
    print("Generator Output Shape:", output.shape)  
    
    disc_output = disc(output)
    print("Discriminator Output Shape:", disc_output.shape) 