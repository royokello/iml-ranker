import argparse
import json
import logging
import random
import numpy as np
import torch
from utils import get_model_by_latest, get_model_by_name, log_print
import os
from PIL import Image
from torchvision import transforms
import glob

def elo_rating(rating1, rating2, outcome, k=32):
    """
    Update Elo ratings based on the outcome of a comparison.

    Args:
        rating1 (float): Current rating of image1.
        rating2 (float): Current rating of image2.
        outcome (str): Outcome of the comparison ("left", "right", "both", "neither").
        k (int, optional): K-factor in Elo rating system. Defaults to 32.

    Returns:
        Tuple[float, float]: Updated ratings for image1 and image2.

    Raises:
        ValueError: If the outcome is not one of the four expected values.
    """
    # Calculate expected scores using the Elo formula
    expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
    expected2 = 1 / (1 + 10 ** ((rating1 - rating2) / 400))

    # Assign actual scores based on the outcome
    if outcome == "left":
        score1, score2 = 1, 0  # Image1 wins
    elif outcome == "right":
        score1, score2 = 0, 1  # Image2 wins
    elif outcome == "both":
        score1, score2 = 1, 1  # Both images are preferred
    elif outcome == "neither":
        score1, score2 = 0, 0  # Neither image is preferred
    else:
        raise ValueError(f"Invalid outcome '{outcome}'. Must be 'left', 'right', 'both', or 'neither'.")

    # Update ratings based on the Elo formula
    new_rating1 = rating1 + k * (score1 - expected1)
    new_rating2 = rating2 + k * (score2 - expected2)

    return new_rating1, new_rating2


def main(root_dir: str, album: str = None, comparisons: int = 10, model_name: str = None):
    """
    Ranks images based on pairwise comparisons using a trained Vision Transformer (ViT) and Elo ratings.

    Args:
        root_dir (str): Root directory containing album folders.
        album (str, optional): Specific album to rank. If None, all albums are processed.
        comparisons (int): Number of comparisons to perform per image.
        model_name (str, optional): Specify model. If None, tries to use 'rank_model' or falls back to the latest model.
    """
    log_filepath = os.path.join(root_dir, 'ranking.log')
    logging.basicConfig(
        filename=log_filepath,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    log_print("Ranking started...")

    # Find the model directory
    models_dir = os.path.join(root_dir, 'models')
    if not os.path.exists(models_dir):
        # Create it if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"Using device: {device}")

    # Try to load the model
    model = None
    
    if model_name:
        # If a specific model name is provided, use that
        model = get_model_by_name(device=device, directory=models_dir, name=model_name)
        if model:
            log_print(f"Using specified model: {model_name}")
    else:
        # First try to use rank_model.pth directly in the root directory (where train.py saves it)
        root_model_path = os.path.join(root_dir, 'rank_model.pth')
        if os.path.exists(root_model_path):
            try:
                from model import ViTRanker
                model = ViTRanker().to(device)
                model.load_state_dict(torch.load(root_model_path, map_location=device))
                log_print(f"Using rank_model.pth from root directory")
            except Exception as e:
                log_print(f"Failed to load rank_model.pth from root directory: {e}")
                model = None
        
        # If not found in the root, try the models subdirectory
        if model is None:
            rank_model_path = os.path.join(models_dir, 'rank_model.pth')
            if os.path.exists(rank_model_path):
                try:
                    from model import ViTRanker
                    model = ViTRanker().to(device)
                    model.load_state_dict(torch.load(rank_model_path, map_location=device))
                    log_print(f"Using rank_model.pth from models subdirectory")
                except Exception as e:
                    log_print(f"Failed to load rank_model.pth from models subdirectory: {e}")
                    model = None
    
    # If model is still None, fall back to latest model
    if model is None:
        model = get_model_by_latest(device=device, directory=models_dir)
        if model:
            log_print("Using latest available model")
    
    if model is None:
        log_print("No model found. Please train a model first.")
        return
    
    model.eval()

    # Transformation for the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjusted to 224x224 for ViT compatibility
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet normalization
                             std=[0.229, 0.224, 0.225]),
    ])

    # Process albums
    if album:
        albums_to_process = [album]
    else:
        # Find all album directories
        albums_to_process = [d for d in os.listdir(root_dir) 
                            if os.path.isdir(os.path.join(root_dir, d)) 
                            and os.path.exists(os.path.join(root_dir, d, 'src'))]
    
    if not albums_to_process:
        log_print("No valid albums found.")
        return
    
    log_print(f"Found {len(albums_to_process)} albums to process.")

    # Process each album
    for current_album in albums_to_process:
        log_print(f"Processing album: {current_album}")
        
        # Get all images from the src directory
        src_dir = os.path.join(root_dir, current_album, 'src_cropped')
        if not os.path.exists(src_dir):
            log_print(f"Source directory not found for album {current_album}, skipping.")
            continue
        
        # Find all image files with multiple extensions
        image_files = []
        for ext in ['.png', '.jpg', '.jpeg']:
            image_files.extend(glob.glob(os.path.join(src_dir, f'*{ext}')))
            image_files.extend(glob.glob(os.path.join(src_dir, f'*{ext.upper()}')))
        
        # Extract just the filenames
        image_files = [os.path.basename(f) for f in image_files]
        
        if len(image_files) < 2:
            log_print(f"Not enough images to rank in album {current_album}.")
            continue

        log_print(f"Found {len(image_files)} images for ranking in album {current_album}.")

        # Load and preprocess all images
        images = {}
        for img_file in image_files:
            img_path = os.path.join(src_dir, img_file)
            try:
                image = Image.open(img_path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0)  # Shape: (1, 3, 224, 224)
                images[img_file] = image_tensor
            except Exception as e:
                log_print(f"Error loading image '{img_file}': {e}")

        # Initialize Elo ratings
        initial_rating = 1500.0
        rankings = {image_file: initial_rating for image_file in images.keys()}

        # Perform pairwise comparisons
        for i, img1 in enumerate(list(images.keys())):
            log_print(f"Comparing image {i + 1}/{len(images)}: {img1}")
            
            img1_tensor = images[img1].to(device)
        
            # Select a random subset of images to compare with img1
            possible_imgs = list(images.keys())
            possible_imgs.remove(img1)
            img_comparisons = random.sample(possible_imgs, min(comparisons, len(possible_imgs)))

            for img2 in img_comparisons:
                img2_tensor = images[img2].to(device)

                with torch.no_grad():
                    try:
                        output = model(img1_tensor, img2_tensor)  # Expected shape: (1, 4)
                        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]  # Shape: (4,)
                        classes = ['left', 'right', 'both', 'neither']

                        # Determine outcome based on highest probability
                        predicted_idx = np.argmax(probabilities)
                        outcome = classes[predicted_idx]

                        log_print(f"  Compared '{img1}' vs '{img2}' - Outcome: {outcome}")

                        # Update Elo ratings
                        rating1 = rankings[img1]
                        rating2 = rankings[img2]

                        new_rating1, new_rating2 = elo_rating(rating1, rating2, outcome)
                        rankings[img1] = new_rating1
                        rankings[img2] = new_rating2

                    except Exception as e:
                        log_print(f"  Error during comparison '{img1}' vs '{img2}': {e}")

        # Sort images based on final ratings
        sorted_rankings = sorted(rankings.items(), key=lambda item: item[1], reverse=True)
        log_print(f"Ranking for album {current_album} completed.")

        # Display rankings
        for rank, (image_file, score) in enumerate(sorted_rankings, 1):
            log_print(f"Rank {rank}: {image_file} with score {score:.2f}")


        # Save rankings to a JSON file
        rankings_path = os.path.join(root_dir, current_album, 'rankings.json')
        try:
            with open(rankings_path, 'w') as f:
                json.dump(sorted_rankings, f, indent=4)
            log_print(f"Rankings saved to '{rankings_path}'.")
        except Exception as e:
            log_print(f"Error saving rankings: {e}")

    log_print("All albums processing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank Images in Album Structure")
    parser.add_argument("-r", "--root_dir", type=str, required=True, help="Root directory containing album folders.")
    parser.add_argument("-a", "--album", type=str, help="Specific album to process. If not provided, all albums are processed.")
    parser.add_argument("-c", "--comparisons", type=int, default=10, help="Number of comparisons per image (default: 10).")
    parser.add_argument("-n", "--model_name", type=str, help="Specify model name. If not provided, the latest model will be used.")
    
    args = parser.parse_args()
    main(args.root_dir, args.album, args.comparisons, model_name=args.model_name)
