# train.py

import os
import argparse
import copy
import shutil
import time
import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import create_data_loaders
from model import ViTRanker
from utils import get_model_by_name

def parse_args():
    """
    Parses command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a Vision Transformer (ViT) for Image Culling.")
    
    parser.add_argument(
        '-r', '--root_dir',
        type=str,
        required=True,
        help='Path to the root directory containing album folders.'
    )
    
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=256,
        help='Number of training epochs (default: 256).'
    )
    
    args = parser.parse_args()
    return args

def save_checkpoint(state, is_best, checkpoint_path, best_model_path):
    """
    Saves the training checkpoint.
    
    Args:
        state (dict): State dictionary containing model state and optimizer state.
        is_best (bool): If True, saves the model as the best model.
        checkpoint_path (str): Path to save the checkpoint.
        best_model_path (str): Path to save the best model.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(state, checkpoint_path)
    
    if is_best:
        # Replace copy.deepcopy with shutil.copyfile
        shutil.copyfile(checkpoint_path, best_model_path)
        print(f"Best model updated: {best_model_path}")

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    
    Args:
        model (nn.Module): The ViTRanker model.
        dataloader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to train on.
    
    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total = 0
    
    for batch_idx, (img1, img2, labels) in enumerate(dataloader):
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.to(device)  # Shape: (batch_size,)
        
        optimizer.zero_grad()
        outputs = model(img1, img2)  # Shape: (batch_size, num_classes)
        loss = criterion(outputs, labels)  # CrossEntropyLoss expects labels of shape (batch_size,)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * img1.size(0)
        total += img1.size(0)
        
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} - Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / total
    return epoch_loss

def validate(model, dataloader, criterion, device):
    """
    Validates the model.
    
    Args:
        model (nn.Module): The ViTRanker model.
        dataloader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to validate on.
    
    Returns:
        Tuple[float, float]: Average validation loss and accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for img1, img2, labels in dataloader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)  # Shape: (batch_size,)
            
            outputs = model(img1, img2)  # Shape: (batch_size, num_classes)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * img1.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += img1.size(0)
    
    epoch_loss = running_loss / total
    accuracy = correct / total
    return epoch_loss, accuracy

def load_all_labels(root_dir):
    """
    Load and combine labels from all album directories.
    
    Args:
        root_dir (str): Path to the root directory containing album folders.
        
    Returns:
        dict: Combined dictionary of all labels from all albums.
    """
    # The dataset.py module now handles this logic
    # We'll use an empty dictionary here and let dataset.py load the CSV files
    return {}

def find_all_images(root_dir):
    """
    Find all source images from all album directories.
    
    Args:
        root_dir (str): Path to the root directory containing album folders.
        
    Returns:
        list: List of paths to all source images.
    """
    all_images = []
    
    # Find all album directories
    for album_dir in os.listdir(root_dir):
        album_path = os.path.join(root_dir, album_dir)
        if os.path.isdir(album_path):
            src_dir = os.path.join(album_path, 'src')
            if os.path.isdir(src_dir):
                # Get all image files
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    image_paths = glob.glob(os.path.join(src_dir, ext))
                    all_images.extend(image_paths)
    
    return all_images

def main():
    # Parse command-line arguments
    args = parse_args()
    
    root_dir = args.root_dir
    epochs = args.epochs
    
    # Fixed parameters
    learning_rate = 1e-4
    batch_size = 8
    num_workers = 2
    scheduler_step = 10
    
    # Define paths
    model_path = os.path.join(root_dir, 'rank_model.pth')
    
    # Make sure models directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find all images (for reporting purposes only)
    print("Checking for images...")
    all_images = find_all_images(root_dir)
    
    if not all_images:
        print("Error: No images found. Ensure that album directories contain src/ folders with image files.")
        return
    
    print(f"Found {len(all_images)} images across all albums.")
    
    # Create data loaders
    try:
        print("Loading and processing labels from CSV files...")
        train_loader, val_loader, test_loader = create_data_loaders(
            root_dir=root_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=(224, 224),  # For ViT compatibility
            train_ratio=0.75,
            val_ratio=0.15,
            test_ratio=0.10,
            random_state=42
        )
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return
    
    # Initialize the model
    print("Initializing the model...")
    if os.path.exists(model_path):
        model = ViTRanker()
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded existing model from {model_path}")
    else:
        model = ViTRanker(num_classes=4, pretrained=True)
    
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # For multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=0.1)
    
    # Training parameters
    best_val_acc = 0.0
    patience = 8  # Number of epochs to wait for improvement
    epochs_no_improve = 0
    
    # Training loop
    print("Starting training...")
    since = time.time()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 10)
        
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f}")
        
        # Step the scheduler
        scheduler.step()
        
        # Check if this is the best model so far
        is_best = val_acc >= best_val_acc
        
        if is_best:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), model_path)
            print(f"New best model found and saved with validation accuracy: {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break
    
    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    
    # Load best model weights
    if 'best_model_wts' in locals():
        model.load_state_dict(best_model_wts)
    
    # Optionally, evaluate on the test set
    print("\nEvaluating on the test set...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    
    # Save the final model
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved at '{model_path}'.")

if __name__ == "__main__":
    main()