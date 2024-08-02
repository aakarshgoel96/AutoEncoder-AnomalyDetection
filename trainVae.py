import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import argparse
import os
from ImageDataset import ImageDataset
from vae_model_2048 import VAE, loss_function
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def check_nan_inf(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"NaN or Inf found in {name}")


def train(model, train_loader, optimizer, epoch, device):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        try:
            recon_batch, mu, logvar = model(data)
        except Exception as e1:
            print(f"Error in model forward pass: {e1}")
            raise
        assert data.shape == recon_batch.shape, f"Shape mismatch: {data.shape} vs {recon_batch.shape}"
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        try:
            loss = loss_function(recon_batch, data, mu, logvar)
        except Exception as e:
            print(f"Error calculating loss: {e}")
            print(f"recon_batch range: [{recon_batch.min()}, {recon_batch.max()}]")
            print(f"data range: [{data.min()}, {data.max()}]")
            raise
        loss.backward()
        try:
            train_loss += loss.item()
        except Exception as e2:
            print(f"Error calculating loss: {e2}")
            raise

        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    return avg_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VAE for Image Reconstruction')
    #parser.add_argument('--train_data_root', type=str, required=True, help='Path to root folder containing training data')
    parser.add_argument('--model_path', type=str, default='vae.pth', help='Path to save the trained model')
    parser.add_argument('--epochs', type=int, default=250, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--plot_path', type=str, default='training_loss_vae.png', help='Path to save the training loss plot')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((2048, 2048)),
        transforms.ToTensor(),  # Converts to [0, 1]
    ])

    image_folder = 'Data/train/ok'
    train_dataset = ImageDataset(image_folder, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE().to(device)
    #model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    training_losses = []
    for epoch in range(1, args.epochs + 1):
        avg_loss = train(model, train_loader, optimizer, epoch, device)
        training_losses.append(avg_loss)
        scheduler.step(avg_loss)  # Pass the average loss to the scheduler
        check_nan_inf(model)  # Add this line to check for NaN/Inf after each epoch
        if epoch % 20 == 0:
            torch.save(model.state_dict(), f'{args.model_path}_epoch_{epoch}.pth')
    
    torch.save(model.state_dict(), args.model_path)

    # Plotting training loss
    plt.plot(training_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(args.plot_path)
    print(f'Training loss plot saved: {args.plot_path}')
