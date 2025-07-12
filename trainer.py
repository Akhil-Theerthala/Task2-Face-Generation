import os
import torch
import numpy as np
import wandb
import dotenv
import torch.nn.functional as F
from torch import nn 
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from faceGAN.dataset import FaceDataset
from faceGAN.networks import Generator, Discriminator, weights_init_normal
from dataclasses import dataclass

os.environ["WANDB_API_KEY"] =dotenv.get_key('.env', 'WANDB_API_KEY')

@dataclass
class TrainingDetails:
    batch_size: int = 64
    val_batch_size: int = 16
    num_epochs: int = 70
    discriminator_learning_rate: float = 3e-4
    generator_learning_rate: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999
    device: str = 'mps' if torch.backends.mps.is_available() else 'cpu'
    train_dir: str = 'data/train'
    val_dir: str = 'data/val'
    
    device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')


class Trainer:
    def __init__(self, config: TrainingDetails):
        self.config = config
        
        if torch.backends.mps.is_available():
            print("Using MPS backend for training.")
        else:
            print("MPS backend not available, using CPU.")
        
        self.train_dataset = FaceDataset(self.config.train_dir)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.val_dataset = FaceDataset(self.config.val_dir)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.config.val_batch_size, shuffle=False)
        
        # Initialize the generator and discriminator
        self.generator = Generator().to(self.config.device)
        self.discriminator = Discriminator().to(self.config.device)
        
        #apply weight initialization
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)
        
        self.criterion = nn.BCEWithLogitsLoss()
        #initialize optimizers
        self.disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.discriminator_learning_rate,
            betas=(self.config.beta1, self.config.beta2)
        )
        
        self.gen_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.generator_learning_rate,
            betas=(self.config.beta1, self.config.beta2)
        )
    
    
    def get_real_loss(self, disc_out):
        return F.softplus(-disc_out).mean()  # Using softplus for real loss

    def get_fake_loss(self, disc_out):
        return F.softplus(disc_out).mean()  # Using softplus for fake loss
    
    def train_step(self, batch, batch_step):
        real_images = batch['image'].to(self.config.device)
        embeddings = batch['embedding'].to(self.config.device)
        
        # Fake images
        fake_images = self.generator(embeddings)
        # fake_images = torch.randn_like(real_images)   # i.i.d 
        
        # condition = ((batch_step//4)%2 == 0)
        #update every other step
        condition = (batch_step % 2 == 0)
        
        if condition:
            # Real images
            disc_real_out = self.discriminator(real_images)
            real_loss = self.get_real_loss(disc_real_out)
            disc_fake_out = self.discriminator(fake_images.detach())
            fake_loss = self.get_fake_loss(disc_fake_out)
    
            disc_loss = real_loss + fake_loss
            disc_loss.backward()
            
            self.disc_optimizer.step()
            self.disc_optimizer.zero_grad()
                
        disc_fake_out_gen = self.discriminator(fake_images)
        gen_loss = self.get_real_loss(disc_fake_out_gen)
        
        gen_loss.backward()
        self.gen_optimizer.step()

        self.gen_optimizer.zero_grad()
        
        return {
            'disc_loss': disc_loss.item() if condition else -1,
            'gen_loss': gen_loss.item(),
            'disc_real_logits_mean': disc_real_out.mean().item() if condition else -1,
            'disc_fake_logits_mean': disc_fake_out.mean().item() if condition else -1
        }
    
    def train(self):
        for epoch in tqdm(range(self.config.num_epochs)):
            epoch_loss = {'disc_loss': 0, 'gen_loss': 0}
            batch_step=0
            old_disc_info = {'loss': -1, "real_mean": -1, "fake_mean": -1}
            for batch in tqdm(self.train_loader):
                
                losses = self.train_step(batch, batch_step=batch_step)
                
                if losses['disc_loss'] == -1:
                    losses['disc_loss'] = old_disc_info['loss']
                    losses['disc_real_logits_mean'] = old_disc_info['real_mean']
                    losses['disc_fake_logits_mean'] = old_disc_info['fake_mean']
                else:
                    old_disc_info['loss'] = losses['disc_loss']
                    old_disc_info['real_mean'] = losses['disc_real_logits_mean']
                    old_disc_info['fake_mean'] = losses['disc_fake_logits_mean']
                    
                batch_step += 1
                epoch_loss['disc_loss'] += losses['disc_loss']
                epoch_loss['gen_loss'] += losses['gen_loss']
                
                wandb.log({
                    'epoch': epoch + 1,
                    'batches/disc_loss': losses['disc_loss'],
                    'batches/gen_loss': losses['gen_loss'],
                    'batches/disc_real_logits_mean': losses['disc_real_logits_mean'],
                    'batches/disc_fake_logits_mean': losses['disc_fake_logits_mean']
                })
            
            # Log the average loss for the epoch
            epoch_loss['disc_loss'] /= len(self.train_loader)
            epoch_loss['gen_loss'] /= len(self.train_loader)
            
            wandb.log({
                'epoch': epoch + 1,
                'epoch/avg_disc_loss': epoch_loss['disc_loss'],
                'epoch/avg_gen_loss': epoch_loss['gen_loss']
            })

            if (epoch+1)%2 == 0:
                self.log_generator_images()
                self.generator.train()
                
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
        
        real_images = (real_images + 1) / 2
        real_images = real_images.clamp(0, 1)
        
        #tensor to numpy
        fake_images = fake_images.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to [batch_size, height, width, channels]
        real_images = real_images.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to [batch_size, height, width, channels]
        
        # Log images to wandb
        wandb.log({
            'generated_images': [wandb.Image(img) for img in fake_images[:num_images]],
            'real_images': [wandb.Image(img) for img in real_images[:num_images]]
        })
    
def main():
    config = TrainingDetails()
    trainer = Trainer(config)
    # Initialize wandb
    wandb.init(
        project="face-generation-gan-mps",
        name = "EXP-8-proof-of-concept",
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