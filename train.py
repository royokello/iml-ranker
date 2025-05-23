# train.py

import csv
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
from model import IMLRankModel

def parse_args():
    """
    Parses command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a Vision Transformer (ViT) for Image Culling.")
    
    parser.add_argument(
        '--project',
        type=str,
        required=True,
        help='Path to the project directory where files are kept.'
    )
    
    parser.add_argument(
        '--stage',
        type=str,
        required=False,
        help='Stage number to use. If not provided, the latest stage will be used. Stages are formatted as "stage_{stage}".'
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
        model (nn.Module): The IMLRankModel model.
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
        model (nn.Module): The IMLRankModel model.
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

def get_latest_stage(project_dir):
    """
    Find the latest stage in the project directory.
    
    Args:
        project_dir (str): Path to the project directory.
        
    Returns:
        str: The latest stage number.
    """
    # Get all directories matching stage_*
    stage_dirs = [d for d in os.listdir(project_dir) if os.path.isdir(os.path.join(project_dir, d)) and d.startswith('stage_')]
    
    if not stage_dirs:
        raise ValueError(f"No stage directories found in {project_dir}. Directories should be named 'stage_{{stage}}'")
    
    # Extract stage numbers and find the maximum
    stage_numbers = []
    for dir_name in stage_dirs:
        try:
            # Extract the stage number after 'stage_'
            stage_num = dir_name.split('_', 1)[1]
            stage_numbers.append(stage_num)
        except (IndexError, ValueError):
            continue
    
    if not stage_numbers:
        raise ValueError(f"Could not parse stage numbers from directories in {project_dir}")
    
    # Return the highest stage number
    return max(stage_numbers)

def find_all_images(project_dir, stage):
    """
    Find all images in the specified stage directory.
    
    Args:
        project_dir (str): Path to the project directory.
        stage (str): The stage identifier.
        
    Returns:
        list: List of paths to all images in the stage directory.
    """
    all_images = []
    
    # Get the stage directory path
    stage_dir = os.path.join(project_dir, f"stage_{stage}")
    
    if not os.path.isdir(stage_dir):
        raise ValueError(f"Stage directory '{stage_dir}' not found")
    
    # Get all image files in the stage directory
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths = glob.glob(os.path.join(stage_dir, ext))
        all_images.extend(image_paths)
    
    return all_images

def main():
    # Parse command-line arguments
    args = parse_args()
    
    project_dir = args.project
    epochs = args.epochs
    
    # Get stage - if not provided, find the latest stage
    if args.stage:
        stage = args.stage
    else:
        try:
            stage = get_latest_stage(project_dir)
            print(f"No stage specified, using latest stage: {stage}")
        except ValueError as e:
            print(f"Error: {e}")
            return
    
    # Fixed parameters
    learning_rate = 1e-4
    batch_size = 8
    num_workers = 2
    scheduler_step = 10
    
    # Define paths
    model_path = os.path.join(project_dir, f"stage_{stage}_rank_model.pth")
    label_csv_path = os.path.join(project_dir, f"stage_{stage}_rank_labels.csv")
    log_file_path = os.path.join(project_dir, f"stage_{stage}_rank_log.csv")

    # Delete existing model and log if it exists
    for path in [model_path, log_file_path]:
        if os.path.exists(path):
            os.remove(path)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if label CSV exists
    if not os.path.exists(label_csv_path):
        print(f"Error: Label CSV not found at {label_csv_path}")
        return
    
    # Find all images (for reporting purposes only)
    print("Checking for images...")
    try:
        all_images = find_all_images(project_dir, stage)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    if not all_images:
        print(f"Error: No images found in stage_{stage} directory. Ensure that the directory contains image files.")
        return
    
    print(f"Found {len(all_images)} images in stage_{stage}.")
    
    # Create data loaders
    try:
        print(f"Loading and processing labels from {label_csv_path}...")
        train_loader, val_loader = create_data_loaders(
            root_dir=os.path.join(project_dir, f"stage_{stage}"),
            labels_csv=label_csv_path,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=(224, 224),  # For ViT compatibility
            train_ratio=0.8,  # Adjusted ratio
            val_ratio=0.2,   # Adjusted ratio
            random_state=42
        )
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return
    
    # Initialize the model
    print("Initializing the model...")
    if os.path.exists(model_path):
        model = IMLRankModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded existing model from {model_path}")
    else:
        model = IMLRankModel(num_classes=4, pretrained=True)
    
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
    
    # Define log file path
    log_file_path = os.path.join(project_dir, f"stage_{stage}_rank_log.csv")

    # Initialize log file (reset and write header)
    with open(log_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy'])

    # Training loop
    print("Starting training...")
    since = time.time()
    
    for epoch in range(epochs):
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Print consolidated epoch results
        print(f"Epoch {epoch + 1}/{epochs}: train_loss: {train_loss:.8f}, val_loss: {val_loss:.8f}, val_acc: {val_acc:.8f}")

        # Log to CSV
        with open(log_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{train_loss:.8f}", f"{val_loss:.8f}", f"{val_acc:.8f}"])
        
        # Step the scheduler
        scheduler.step()
        
        # Calculate a dynamic threshold based on 10% of the smaller loss
        threshold = 0.1 * min(train_loss, val_loss)
        # Check if this is the best model (accuracy increasing or same accuracy but train and val loss gap isnt too big)
        is_best = (val_acc > best_val_acc) or (val_acc == best_val_acc and abs(train_loss - val_loss) < threshold)
        
        if is_best:
            best_val_acc = val_acc
            epochs_no_improve = 0
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
    
    print(f"Final model saved at '{model_path}'.")
    print(f"Stage: {stage}")
    print(f"Project: {project_dir}")

if __name__ == "__main__":
    main()