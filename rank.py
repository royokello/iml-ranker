import argparse
import csv
import random
import numpy as np
import torch
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


def get_latest_stage(project_dir):
    """
    Find the latest stage in the project directory.
    
    Args:
        project_dir (str): Path to the project directory.
        
    Returns:
        int: The latest stage number.
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
            stage_num = int(dir_name.split('_', 1)[1])
            stage_numbers.append(stage_num)
        except (IndexError, ValueError):
            continue
    
    if not stage_numbers:
        raise ValueError(f"Could not parse stage numbers from directories in {project_dir}")
    
    # Return the highest stage number
    return max(stage_numbers)


def main(project_dir: str, stage: int = None, comparisons: int = 10, batch_size: int = 8, verbose: bool = False):
    """
    Ranks images based on pairwise comparisons using a trained Vision Transformer (ViT) and Elo ratings.

    Args:
        project_dir (str): Project directory containing stage folders and models.
        stage (int, optional): Specific stage to use. If None, the latest stage will be used.
        comparisons (int): Number of comparisons to perform per image.
    """
    # If stage is not provided, use the latest stage
    if stage is None:
        try:
            stage = get_latest_stage(project_dir)
            print(f"Using latest stage: {stage}")
        except ValueError as e:
            print(f"Error finding latest stage: {e}")
            return
    
    
    print(f"Ranking started for stage_{stage}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model for the specified stage
    model_path = os.path.join(project_dir, f"stage_{stage}_rank_model.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    try:
        from model import IMLRankModel
        model = IMLRankModel().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        return
    
    # Set up the stage directory where images are stored
    stage_dir = os.path.join(project_dir, f"stage_{stage}")
    if not os.path.exists(stage_dir):
        print(f"Error: Stage directory not found at {stage_dir}")
        return
    
    model.eval()

    # Transformation for the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjusted to 224x224 for ViT compatibility
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet normalization
                             std=[0.229, 0.224, 0.225]),
    ])

    # Process the stage directory
    albums_to_process = [f"stage_{stage}"]
    
    if not albums_to_process:
        print("No valid albums found.")
        return
    
    print(f"Found {len(albums_to_process)} albums to process.")

    # Process the stage directory
    current_album = f"stage_{stage}"
    print(f"Processing stage: {current_album}")
    
    # Set up source directory
    src_dir = stage_dir
            
    if not os.path.exists(src_dir):
        print(f"Stage directory not found at {src_dir}")
        return
    
    # Get all entry names (files and directories) from the source directory.
    entry_names = os.listdir(src_dir)

    # Load and preprocess all actual image files from these entries
    images = {}
    for entry_name in entry_names:
        full_path = os.path.join(src_dir, entry_name)
        try:
            # Attempt to open as an image. This will fail for directories or non-image files.
            image = Image.open(full_path).convert("RGB")
            # If successful, it's an image we can process.
            image_tensor = transform(image).unsqueeze(0)  # Shape: (1, 3, 224, 224)
            img_basename = os.path.basename(full_path)
            images[img_basename] = image_tensor
        except (IOError, OSError) as e: # Catches IsADirectoryError, FileNotFoundError, UnidentifiedImageError from PIL etc.
            if verbose: # verbose is an argument to main()
                print(f"Skipping '{entry_name}': Not a loadable image file (or is a directory). Details: {e}")
            # Silently skip if not verbose, or if it's an expected non-image file type.
    
    print(f"Found {len(images)} valid images for ranking in stage {stage}.")
    if len(images) < 2:
        print(f"Not enough valid images to rank in stage {stage} (found {len(images)}, need at least 2).")
        return

    # Initialize Elo ratings
    initial_rating = 1500.0
    rankings = {image_file: initial_rating for image_file in images.keys()}

    # Perform pairwise comparisons
    for i, img1_name in enumerate(list(images.keys())):
        print(f"{i + 1}/{len(images)}")
        
        img1_tensor_single = images[img1_name].to(device) # This is (1, C, H, W)
    
        # Select a random subset of images to compare with img1_name
        possible_comparison_img_names = list(images.keys())
        possible_comparison_img_names.remove(img1_name)
        # Ensure we have enough images for the requested number of comparisons
        num_to_sample = min(comparisons, len(possible_comparison_img_names))
        if num_to_sample == 0:
            continue
            
        selected_comparison_img_names = random.sample(possible_comparison_img_names, num_to_sample)

        # Process comparisons in batches
        for batch_start_idx in range(0, len(selected_comparison_img_names), batch_size):
            batch_end_idx = min(batch_start_idx + batch_size, len(selected_comparison_img_names))
            current_batch_img_names = selected_comparison_img_names[batch_start_idx:batch_end_idx]
            
            if not current_batch_img_names:
                continue

            # Prepare batch tensors
            # img1_tensor_single is (1, C, H, W), repeat it for the batch
            # For a batch of size N, img1_batch should be (N, C, H, W)
            img1_batch = img1_tensor_single.repeat(len(current_batch_img_names), 1, 1, 1)
            
            # img2_batch should also be (N, C, H, W)
            img2_tensors_list = [images[name].to(device) for name in current_batch_img_names]
            img2_batch = torch.cat(img2_tensors_list, dim=0)

            with torch.no_grad():
                try:
                    # Model expects two batches of images: (N, C, H, W), (N, C, H, W)
                    # Output should be (N, 4)
                    batch_outputs = model(img1_batch, img2_batch)
                    batch_probabilities = torch.softmax(batch_outputs, dim=1).cpu().numpy() # Shape: (N, 4)
                    
                    classes = ['left', 'right', 'both', 'neither']

                    for batch_idx, img2_name in enumerate(current_batch_img_names):
                        probabilities = batch_probabilities[batch_idx] # Shape: (4,)
                        predicted_idx = np.argmax(probabilities)
                        outcome = classes[predicted_idx]

                        if verbose:
                            print(f"  Compared '{img1_name}' vs '{img2_name}' - Outcome: {outcome} (Probs: {probabilities})")

                        # Update Elo ratings
                        rating1 = rankings[img1_name]
                        rating2 = rankings[img2_name]

                        new_rating1, new_rating2 = elo_rating(rating1, rating2, outcome)
                        rankings[img1_name] = new_rating1
                        rankings[img2_name] = new_rating2

                except Exception as e:
                    print(f"  Error during batched comparison for '{img1_name}' vs batch {current_batch_img_names}: {e}")
                    # Optionally, decide if you want to skip the rest of this batch or img1_name
                    # For now, we'll log and continue with the next batch/image
                    break # Break from inner loop (batches for current img1_name)

    # Sort images based on final ratings
    sorted_rankings = sorted(rankings.items(), key=lambda item: item[1], reverse=True)
    print(f"Ranking for stage {stage} completed.")

    # Display rankings
    if verbose:
        print("\nFinal Rankings:")
        for rank, (image_file, score) in enumerate(sorted_rankings, 1):
            print(f"  Rank {rank}: {image_file} (Score: {score:.2f})")


    # Save rankings to a CSV file
    rankings_path = os.path.join(project_dir, f"stage_{stage}_rankings.csv")
            
    try:
        with open(rankings_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["image", "score"])  # Header row
            for rank, (image_file, score) in enumerate(sorted_rankings, 1):
                writer.writerow([image_file, f"{score:.2f}"])
        print(f"\nRankings saved to '{rankings_path}'.")
    except Exception as e:
        print(f"\nError saving rankings to CSV: {e}")

    print("Stage processing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank Images using trained model")
    parser.add_argument("--project", type=str, required=True, help="Project directory path where stage folders and models are stored.")
    parser.add_argument("--stage", type=int, help="Stage to use. If not provided, the latest stage will be used. Stages are formatted as 'stage_{stage}'.")
    parser.add_argument("--comparisons", type=int, default=8, help="Number of comparisons per image (default: 8).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for comparisons (default: 8).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    
    args = parser.parse_args()
    main(args.project, args.stage, args.comparisons, args.batch_size, args.verbose)
