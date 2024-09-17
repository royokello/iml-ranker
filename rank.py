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


def main(working_dir: str, comparisons: int, model_name: str = None):
    """
    Ranks images based on pairwise comparisons using a trained Custom ResNet and Elo ratings.

    Args:
        working_dir (str): Working directory containing 'ranker' and 'cropper' subdirectories.
        comparisons (int): Number of comparisons to perform per image.
        model_name (str, optional): Specify model. If None, loads the latest model.
    """
    ranker_dir = os.path.join(working_dir, 'ranker')
    log_filepath = os.path.join(ranker_dir, 'ranking.log')
    logging.basicConfig(
        filename=log_filepath,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    log_print("Ranking started...")

    cropper_output_dir = os.path.join(working_dir, 'cropper', 'output', '256p')
    models_dir = os.path.join(ranker_dir, 'models')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"Using device: {device}")

    if model_name:
        model = get_model_by_name(device=device, directory=models_dir, name=model_name)
    else:
        model = get_model_by_latest(device=device, directory=models_dir)
    model.eval()

    # Load and preprocess all images
    image_files = [f for f in os.listdir(cropper_output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(image_files) < 2:
        log_print("Not enough images to rank.")
        return

    log_print(f"Found {len(image_files)} images for ranking.")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Adjust based on your model's requirements
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Use the same normalization as training
                                std=[0.229, 0.224, 0.225]),
    ])

    images = {}
    for img_file in image_files:
        img_path = os.path.join(cropper_output_dir, img_file)
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0)  # Shape: (1, 3, 224, 224)
            images[img_file] = image_tensor
        except Exception as e:
            log_print(f"Error loading image '{img_file}': {e}")

    initial_rating = 1500.0
    rankings = {image_file: initial_rating for image_file in image_files}

    for i, img1 in enumerate(image_files):
        log_print(f"Comparing image {i + 1}/{len(image_files)}: {img1}")
        
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
    log_print("Ranking completed.")

    # Display rankings
    for rank, (image_file, score) in enumerate(sorted_rankings, 1):
        log_print(f"Rank {rank}: {image_file} with score {score:.2f}")

    # Save rankings to a JSON file
    rankings_path = os.path.join(ranker_dir, 'rankings.json')
    try:
        with open(rankings_path, 'w') as f:
            json.dump(sorted_rankings, f, indent=4)
        log_print(f"Rankings saved to '{rankings_path}'.")
    except Exception as e:
        log_print(f"Error saving rankings: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank Images")
    parser.add_argument("-w", "--working_dir", type=str, required=True, help="Working Directory in ILM Format.")
    parser.add_argument("-c", "--comparisons", type=int, required=True, help="No. of comparisons per image.")
    parser.add_argument("-n", "--model_name", type=str, required=False, help="Specify model.")
    
    args = parser.parse_args()
    main(args.working_dir, args.comparisons, model_name=args.model_name)
