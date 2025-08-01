import torch
import numpy as np

def convert_dataset(dataset):
    images = torch.stack([torch.tensor(np.array(img).astype('float32')) / 255.0 for img, _ in dataset])
    labels = torch.tensor([label for _, label in dataset])
    
    mean, std = 0.2860, 0.3530  # Precalculados para FashionMNIST
    images = (images - mean) / std
    
    return images.to('cuda'), labels.unsqueeze(1).to('cuda')
