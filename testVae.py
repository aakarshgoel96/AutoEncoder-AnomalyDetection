import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from ImageDataset import ImageDataset
from PIL import Image

from vae_model_2048 import VAE, loss_function


def save_reconstructed_images(recon_images, filenames, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    recon_images = recon_images.permute(0, 2, 3, 1).cpu().numpy()
    for recon_image, filename in zip(recon_images, filenames):
        recon_image = (recon_image * 255).astype(np.uint8)
        image = Image.fromarray(recon_image)
        image.save(os.path.join(output_dir, filename))

def test_vae(model, test_loader, device, anomaly_scores, output_dir):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            recon, mu, logvar = model(data)
            loss = loss_function(recon, data, mu, logvar)
            test_loss += loss.item()
            recon_error = ((data - recon) ** 2).mean(dim=[1, 2, 3]).cpu().numpy()
            anomaly_scores.extend(recon_error)
            
            filenames = [f'reconstructed_{batch_idx}_{i}.png' for i in range(data.size(0))]
            save_reconstructed_images(recon, filenames, output_dir)
            
    return test_loss / len(test_loader.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test VAE for Image Reconstruction and Anomaly Detection')
    parser.add_argument('--test_data_root', type=str, default='Data/test2', help='Path to root folder containing test data categories')
    parser.add_argument('--model_path', type=str, default='vae.pth_epoch_40.pth', help='Path to saved model')
    parser.add_argument('--plot_path', type=str, default='reconstruction_errors_vae.png', help='Path to save merged reconstruction error plot')
    parser.add_argument('--output_dir', type=str, default='reconstructed_images_vae', help='Directory to save reconstructed images')
    args = parser.parse_args()

    # Data transformations and loaders
    transform = transforms.Compose([
        transforms.Resize((2048, 2048)),
        transforms.ToTensor()
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE().to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    reconstruction_errors = {}
    categories = [category for category in os.listdir(args.test_data_root) if os.path.isdir(os.path.join(args.test_data_root, category))]
    colormap = plt.cm.get_cmap('tab20', len(categories))
    category_colors = {category: colormap(i) for i, category in enumerate(categories)}

    for category in categories:
        if (category in ["err_geometrie", "err_geometrie_laser"]):
            continue
        category_path = os.path.join(args.test_data_root, category)
        test_dataset = ImageDataset(category_path, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        anomaly_scores = []
        output_category_dir = os.path.join(args.output_dir, category)
        test_loss = test_vae(model, test_loader, device, anomaly_scores, output_category_dir)
        print(f'Test loss for category {category}: {test_loss:.4f}')
        reconstruction_errors[category] = anomaly_scores

        # Plotting reconstruction errors
        for idx, score in enumerate(anomaly_scores):
            plt.scatter(idx, score, color=category_colors[category], label=category if idx == 0 else "")

    # Save the merged plot
    plt.xlabel('Image Index')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Errors for Test Images Across Categories')
    plt.legend()
    plt.grid(True)
    plt.savefig(args.plot_path)
    print(f'Reconstruction error plot saved: {args.plot_path}')
