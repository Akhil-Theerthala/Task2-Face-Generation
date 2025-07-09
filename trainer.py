import os
import torch
import numpy as np
import wandb
from torch import nn 
from torch.utils.data import DataLoader
from faceGAN.dataset import FaceDataset
from faceGAN.networks import Generator, Discriminator
from dataclasses import dataclass

os.environ["WANDB_API_KEY"] ="api_key"


@dataclass
class TrainingDetails:
    batch_size: int = 32
    val_batch_size: int = 16
    num_epochs: int = 30
    learning_rate: float = 3e-4
    beta1: float = 0.5
    beta2: float = 0.999
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dir: str = 'data/train'
    val_dir: str = 'data/val'
    
    device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')


class Trainer:
    def __init__(self, config: TrainingDetails):
        self.config = config
        
        self.train_dataset = FaceDataset(self.config.train_dir)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.val_dataset = FaceDataset(self.config.val_dir)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.config.val_batch_size, shuffle=False)
        
        # Initialize the generator and discriminator
        self.generator = Generator().to(self.config.device)
        self.discriminator = Discriminator().to(self.config.device)
        
        
        #initialize optimizers
        self.disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2)
        )
        
        self.gen_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2)
        )
    
    def get_real_loss(self, disc_out):
        batch_size = disc_out.size(0)
        real_labels = torch.ones(batch_size, 1, device=self.config.device)
        real_labels = real_labels*0.9
        
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(disc_out.squeeze(), real_labels)
        return loss

    def get_fake_loss(self, disc_out):
        batch_size = disc_out.size(0)
        fake_labels = torch.zeros(batch_size, 1, device=self.config.device)
        
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(disc_out.squeeze(), fake_labels)
        return loss
    
    def train_step(self, batch):
        real_images = batch['image'].to(self.config.device)
        embeddings = batch['embedding'].to(self.config.device)
        
        # Train Discriminator
        self.disc_optimizer.zero_grad()
        
        # Real images
        disc_real_out = self.discriminator(real_images)
        real_loss = self.get_real_loss(disc_real_out)
        
        # Fake images
        fake_images = self.generator(embeddings)
        disc_fake_out = self.discriminator(fake_images.detach())
        fake_loss = self.get_fake_loss(disc_fake_out)
        
        disc_loss = real_loss + fake_loss
        disc_loss.backward()
        self.disc_optimizer.step()
        
        # Train Generator
        self.gen_optimizer.zero_grad()
        
        disc_fake_out_gen = self.discriminator(fake_images)
        gen_loss = self.get_real_loss(disc_fake_out_gen)
        
        gen_loss.backward()
        self.gen_optimizer.step()
        
        return {
            'disc_loss': disc_loss.item(),
            'gen_loss': gen_loss.item(),
            'fake_images': fake_images
        }
    
    def train(self):
        for epoch in range(self.config.num_epochs):
            
            self.discriminator.train()
            self.generator.train()
            
            epoch_loss = {'disc_loss': 0, 'gen_loss': 0}
            for batch in self.train_loader:
                losses = self.train_step(batch)
                epoch_loss['disc_loss'] += losses['disc_loss']
                epoch_loss['gen_loss'] += losses['gen_loss']
                
                wandb.log({
                    'epoch': epoch + 1,
                    'batches/disc_loss': losses['disc_loss'],
                    'batches/gen_loss': losses['gen_loss']
                })
            
            # Log the average loss for the epoch
            epoch_loss['disc_loss'] /= len(self.train_loader)
            epoch_loss['gen_loss'] /= len(self.train_loader)
            
            wandb.log({
                'epoch': epoch + 1,
                'epoch/avg_disc_loss': epoch_loss['disc_loss'],
                'epoch/avg_gen_loss': epoch_loss['gen_loss']
            })

            if (epoch+1)%5 == 0:
                self.log_generator_images()
        # Save the model checkpoints
        torch.save(self.generator.state_dict(), 'generator.pth')
        torch.save(self.discriminator.state_dict(), 'discriminator.pth')
    
    def log_generator_images(self, num_images=16):
        #get a val_batch
        val_batch = next(iter(self.val_loader))
        real_images = val_batch['image'].to(self.config.device)
        embeddings = val_batch['embedding'].to(self.config.device)
        
        self.generator.eval()
        with torch.inference_mode():
            fake_images = self.generator(embeddings)
        fake_images = (fake_images + 1) / 2
        fake_images = fake_images.clamp(0, 1)
        
        #tensor to numpy
        fake_images = fake_images.cpu().numpy()
        real_images = real_images.cpu().numpy()
        
        # Log images to wandb
        grid = wandb.Image(fake_images, caption="Generated Images")
        wandb.log({"generated_images": grid})
        
        #real images
        real_grid = wandb.Image(real_images, caption="Real Images")
        wandb.log({"real_images": real_grid})
    
def main():
    config = TrainingDetails()
    trainer = Trainer(config)
    # Initialize wandb
    wandb.init(
        project="face-generation-gan",
        name = "face-generation-gan-training",
        dir="./wandb_logs",
        notes="Training a GAN for face generation using a simple deconv generator and a fastvit discriminator.",
        config=config.__dict__,  
    )
    
    # Log the model architecture
    wandb.watch(trainer.generator, log="all")
    wandb.watch(trainer.discriminator, log="all")
    
    
    # Start training
    print("Starting training...")
    trainer.train()

if __name__ == "__main__":
    main()
    wandb.finish()  # Finish the wandb run