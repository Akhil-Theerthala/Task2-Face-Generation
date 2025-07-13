# dataset generation
import os
import torch
import timm
import random
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

random.seed(42)  # For reproducibility

class FaceDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        
        self.embedding_model = timm.create_model("hf_hub:gaunernst/vit_tiny_patch8_112.arcface_ms1mv3", pretrained=True).eval()
        self.data_config = timm.data.resolve_data_config(self.embedding_model.pretrained_cfg)
        self.transform = timm.data.create_transform( **self.data_config, is_training=False)
        if not self.image_files:
            raise ValueError(f"No images found in directory: {image_dir}")
        
        self.flip_prob = 0.3
        self.to_latent = transforms.Compose([
            transforms.RandomResizedCrop(128, scale=(0.9,1.0)),
            transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
            ])
        
    
    def get_embedding(self, image):
        with torch.inference_mode():
            image = self.transform(image).unsqueeze(0)
            embedding = self.embedding_model(image)
            embedding = F.normalize(embedding, dim=1)
        return embedding.squeeze(0) # From [1, 512] to [512]
            
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        pil_image = Image.open(img_path).convert('RGB')   

        if random.random() < self.flip_prob:
            pil_image = transforms.functional.hflip(pil_image)
        
        embedding = self.get_embedding(pil_image)
        
        # #setting up an unconditional_base_line
        # embedding = torch.randn(1,512)
        
        image = self.to_latent(pil_image)
        return {
            'image': image,
            'embedding': embedding
        }
    

if __name__ == "__main__":    
    train_dataset = FaceDataset('data/train')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print(train_dataset.__len__(), "training images found.")
    print("TinyViT Transform:", train_dataset.transform)
    batch = next(iter(train_loader))
    print("Batch Size:", batch['image'].shape)
    print("Batch Embedding Shape:", batch['embedding'].shape)
    