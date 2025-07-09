# dataset generation
import os
import torch
import timm
import random
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from timm.data.transforms import MaybeToTensor, RandomResizedCropAndInterpolation

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
        
        self.to_latent = transforms.Compose([
            RandomResizedCropAndInterpolation(size=(128, 128), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation='bicubic'),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=None),
            MaybeToTensor(),
            transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
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
        
        flip_prob=0.2
        if random.random() < flip_prob:
            print(f"Flipping image: {self.image_files[idx]}")
            pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
        
        crop_prob=0.15
        if random.random() < crop_prob:
            print(f"Cropping image: {self.image_files[idx]}")
            width, height = pil_image.size
            left = int(width * 0.05)
            top = int(height * 0.05)
            right = int(width * 0.95)
            bottom = int(height * 0.95)
            pil_image = pil_image.crop((left, top, right, bottom))
        
        embedding = self.get_embedding(pil_image)
        
        image = self.to_latent(pil_image)
        
        return {
            'image': image,
            'embedding': embedding
        }
    

if __name__ == "__main__":    
    train_dataset = FaceDataset('data/train')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print(train_dataset.__len__(), "training images found.")