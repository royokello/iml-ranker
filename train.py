# train.py

import os
import argparse
import copy
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import create_data_loaders
from model import CustomResNet  # Updated import to use CustomResNet
from utils import get_model_by_name  # Ensure this utility is compatible with CustomResNet

def parse_args():
    """
    Parses command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a Custom ResNet for Image Preference Learning.")
    
    parser.add_argument(
        '-w', '--working_dir',
        type=str,
        required=True,
        help='Path to the working directory containing ranker and cropper directories.'
    )
    
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=64,
        help='Number of training epochs (default: 64).'
    )
    
    parser.add_argument(
        '-c', '--num_workers',
        type=int,
        default=8,
        help='Number of worker processes for data loading (default: 8).'
    )
    
    parser.add_argument(
        '-b', '--batch_size',
        type=int,
        default=32,
        help='Batch size for training (default: 32).'
    )
    
    parser.add_argument(
        '-lr', '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate for the optimizer (default: 1e-4).'
    )
    
    parser.add_argument(
        '-s', '--scheduler_step',
        type=int,
        default=10,
        help='Step size for learning rate scheduler (default: 10 epochs).'
    )
    
    parser.add_argument(
        '-cr', '--checkpoint_resume',
        type=str,
        default=None,
        help='Path to a checkpoint file to resume training.'
    )

    parser.add_argument(
        '-cf', '--checkpoint_freq',
        type=int,
        default=8,
        help='Checkpoint frequency in epochs (default: 8).'
    )
    
    args = parser.parse_args()
    return args

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    """
    Saves the training checkpoint.
    
    Args:
        state (dict): State dictionary containing model state and optimizer state.
        is_best (bool): If True, saves the model as the best model.
        checkpoint_dir (str): Directory to save the checkpoint.
        filename (str): Filename for the checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        # Replace copy.deepcopy with shutil.copyfile
        shutil.copyfile(checkpoint_path, best_path)
        print(f"Best model updated: {best_path}")

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    
    Args:
        model (nn.Module): The Custom ResNet model.
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
        model (nn.Module): The Custom ResNet model.
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

def main():
    # Parse command-line arguments
    args = parse_args()
    
    working_dir = args.working_dir
    epochs = args.epochs
    num_workers = args.num_workers
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    scheduler_step = args.scheduler_step
    checkpoint_resume = args.checkpoint_resume
    checkpoint_freq = args.checkpoint_freq
    
    # Define paths
    ranker_dir = os.path.join(working_dir, 'ranker')
    cropper_output_dir = os.path.join(working_dir, 'cropper', 'output', '256p')
    models_dir = os.path.join(ranker_dir, 'models')
    labels_file = os.path.join(ranker_dir, 'labels.json')
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Loading data...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            image_dir=cropper_output_dir,
            labels_file=labels_file,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=(256, 256),
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
    if checkpoint_resume:
        model = get_model_by_name(device=device, directory=models_dir, name=checkpoint_resume)
    else:
        model = CustomResNet()
    
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Updated for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=0.1)
    
    # Optionally resume from a checkpoint
    best_val_acc = 0.0
    start_epoch = 0

    patience = 8  # Number of epochs to wait for improvement
    epochs_no_improve = 0

    # Training loop
    print("Starting training...")
    since = time.time()
    
    for epoch in range(start_epoch, epochs):
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
            checkpoint_path = os.path.join(models_dir, "best.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model found and saved with validation accuracy: {best_val_acc:.4f}")
        
        else:
            epochs_no_improve += 1

        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(models_dir, f"{int(time.time())}_e={epoch + 1}")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1}.")

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
    final_model_path = os.path.join(models_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at '{final_model_path}'.")

if __name__ == "__main__":
    main()