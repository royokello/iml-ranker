from torch.utils.data import Dataset
from PIL import Image
import os
import json
import torch

class ImageDataset(Dataset):
    def __init__(self, images_dir, labels_file, transform):
        self.images_dir = images_dir
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)
        self.image_pairs = list(self.labels.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        pair = self.image_pairs[idx]
        image1_name, image2_name = pair.split('_')
        
        img1_path = os.path.join(self.images_dir, f"{image1_name}.png")
        img2_path = os.path.join(self.images_dir, f"{image2_name}.png")
        
        image1 = Image.open(img1_path).convert("RGB")
        image2 = Image.open(img2_path).convert("RGB")

        label = self.labels[pair]

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        
        return image1, image2, torch.tensor(float(label), dtype=torch.float32)
