import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ImageDataset
from model import ComparisonNet
from utils import generate_model_name, get_model_by_name, log_print, setup_logging
import torchvision.transforms as transforms

def train(working_dir: str, epochs: int, checkpoint: int, base_model: str|None):
    ranker_dir = os.path.join(working_dir, 'ranker')
    setup_logging(ranker_dir)
    log_print("training started ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"using {device}")

    # Setup directories
    models_dir = os.path.join(ranker_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    images_dir = os.path.join(working_dir, 'cropper', 'output', '256p')
    labels_file = os.path.join(ranker_dir, 'labels.json')

    # Initialize the model, optimizer, and loss function
    if base_model:
        model = get_model_by_name(device=device, directory=models_dir, name=base_model)
    else:
        model = ComparisonNet().to(device)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # type: ignore
    criterion = nn.BCEWithLogitsLoss()

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Initialize dataset and dataloader
    dataset = ImageDataset(images_dir=images_dir, labels_file=labels_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (image1, image2, labels) in enumerate(dataloader):
            image1, image2, labels = image1.to(device), image2.to(device), labels.to(device)

            # Forward pass
            outputs = model(image1, image2).squeeze()
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Log epoch loss
        log_print(f"epoch [{epoch+1}/{epochs}], average loss: {epoch_loss/len(dataloader):.4f}")
        
        if (epoch + 1) % checkpoint == 0:
            checkpoint_name = generate_model_name(base_model, len(dataset), epoch + 1)
            checkpoint_path = os.path.join(models_dir, f"{checkpoint_name}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            log_print(f"checkpoint saved: {checkpoint_path}")

    log_print("Training completed.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("-w", "--working_dir", type=str, required=True, help="Directory where the training data is located.")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="Number of epochs to train.")
    parser.add_argument("-c", "--checkpoint", type=int, required=True, help="Number of checkpoints to save.")
    parser.add_argument("-b", "--base_model", type=str, help="Name of the base model if continuing training, or None if starting from scratch.")
    
    args = parser.parse_args()
    
    train(
        working_dir=args.working_dir,
        epochs=args.epochs,
        checkpoint=args.checkpoint,
        base_model=args.base_model
    )