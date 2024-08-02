import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from Autoencoder import Autoencoder
from ImageDataset import ImageDataset
import argparse
import numpy as np

# Define a function to load the trained model
def load_model(model_path):
    model = Autoencoder()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model



# Define the function to calculate MSE loss for each image
def calculate_mse(original, reconstructed):
    return np.mean(np.square(original - reconstructed))

def test_model(model, test_loader, device, category, color, reconstruction_errors):
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        with open("scatter_test_data.txt", 'a') as f:
            for idx, images in enumerate(test_loader):
                images = images.to(device)
                outputs = model(images)
                
                for i in range(images.size(0)):
                    """ orig_img = transforms.ToPILImage()(images[i].cpu())
                    recon_img = transforms.ToPILImage()(outputs[i].cpu())
                    
                    orig_img.save(f'{save_path}/original_{idx * test_loader.batch_size + i}.png')
                    recon_img.save(f'{save_path}/reconstructed_{idx * test_loader.batch_size + i}.png') """
                
                    # Calculate MSE reconstruction error
                    mse_error = calculate_mse(images[i].cpu().numpy(), outputs[i].cpu().numpy())
                    reconstruction_errors[category].append(mse_error)
                
                    # Plot the individual image with a different color for each category
                    plt.scatter(idx, mse_error, color=color, label=category if idx == 0 else "")
                    f.write(f"{category}\t{idx}\t{mse_error} \n")
                

    
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Autoencoder on Image Dataset')
    parser.add_argument('--test_data_root', type=str, required=True, help='Path to test data folder')
    parser.add_argument('--model_path', type=str, default='autoencoder.pth', help='Path to saved model')
    parser.add_argument('--save_path', type=str, default='reconstructed_images_test', help='Path to save reconstructed images')
    parser.add_argument('--plot_path', type=str, default='reconstruction_errors_test.png', help='Path to save reconstruction error plot')
    args = parser.parse_args()
    
    # Define test transforms and dataloader
    test_transform = transforms.Compose([
        transforms.Resize((2048, 2048)),
        transforms.ToTensor()
    ])
    
    # Load the trained model and test it
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(args.model_path)

    reconstruction_errors = {}

    categories = [category for category in os.listdir(args.test_data_root) if os.path.isdir(os.path.join(args.test_data_root, category))]
    colormap = plt.cm.get_cmap('tab20', len(categories))
    category_colors = {category: colormap(i) for i, category in enumerate(categories)}

    # Iterate over each category in the root test data folder
    with open("scatter_test_data.txt", 'w') as f:
        f.write("label\tIndex\Error \n")

    for category in categories:
        if (category in ["err_geometrie", "err_geometrie_laser"]):
            continue
        category_path = os.path.join(args.test_data_root, category)
        if os.path.isdir(category_path):
            print(f'Testing category: {category}')
            test_dataset = ImageDataset(category_path, transform=test_transform)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            
            reconstruction_errors[category] = []
            test_model(model, test_loader, device, category, category_colors[category], reconstruction_errors)
    
    # Plot merged reconstruction errors
    plt.xlabel('Image Index')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.title('Reconstruction Errors for Test Images Across Categories')
    plt.legend()
    plt.grid(True)
    plt.savefig(args.plot_path)
    print(f'Reconstruction error plot saved: {args.plot_path}')
    #test_model(model, test_loader, device, args.save_path, args.plot_path)