import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import transforms
from PIL import Image
from Autoencoder import Autoencoder
import argparse
import numpy as np

class SingleImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.image_files[idx]

def load_model(model_path, device):
    model = Autoencoder()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def calculate_mse(original, reconstructed):
    return np.mean(np.square(original - reconstructed))

def test_model(model, test_loader, device, output_file):
    model.to(device)
    model.eval()
    
    with torch.no_grad(), open(output_file, 'w') as f:
        f.write("Image Name\tReconstruction Error (MSE)\n")
        for images, image_names in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            for i in range(images.size(0)):
                mse_error = calculate_mse(images[i].cpu().numpy(), outputs[i].cpu().numpy())
                f.write(f"{image_names[i]}\t{mse_error:.6f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Autoencoder on Image Dataset')
    parser.add_argument('--test_data_root', type=str, required=True, help='Path to test data folder')
    parser.add_argument('--model_path', type=str, default='autoencoder.pth', help='Path to saved model')
    parser.add_argument('--output_file', type=str, default='reconstruction_errors_new.txt', help='Path to save reconstruction errors')
    args = parser.parse_args()
    
    test_transform = transforms.Compose([
        transforms.Resize((2048, 2048)),
        transforms.ToTensor()
    ])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(args.model_path, device)

    test_dataset = SingleImageDataset(args.test_data_root, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    test_model(model, test_loader, device, args.output_file)
    print(f'Reconstruction errors saved to: {args.output_file}')