import argparse
import json
import random
import torch
from utils import get_model_by_latest, log_print
import os
from predict import predict
from PIL import Image
from torchvision import transforms

def elo_rating(rating1, rating2, outcome, k=32):
    """
    Update Elo ratings based on the outcome of a comparison.
    """
    expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
    expected2 = 1 / (1 + 10 ** ((rating1 - rating2) / 400))

    new_rating1 = rating1 + k * (outcome - expected1)
    new_rating2 = rating2 + k * ((1 - outcome) - expected2)

    return new_rating1, new_rating2

def rank(working_dir: str, comparisons: int):
    """
    """
    print("ranking started ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device}")

    models_path = os.path.join(working_dir, 'ranker', 'models')
    model = get_model_by_latest(device=device, directory=models_path)
    model.eval()

    images_path = os.path.join(working_dir, 'cropper', 'output', '256p')
    image_files = [f for f in os.listdir(images_path) if f.endswith('.png')]

    if len(image_files) < 2:
        print("Not enough images to rank.")
        return
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    images = {}
    for img_file in image_files:
        img_path = os.path.join(images_path, img_file)
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        images[img_file] = image_tensor

    initial_rating = 1500.0
    rankings = {image_file: initial_rating for image_file in image_files}

    for i, img1 in enumerate(image_files):
        print(f" * {i} / {len(image_files)}")
        
        img1_tensor = images[img1].to(device)
    
        img_comparisons = random.sample(image_files, min(comparisons, len(image_files) - 1))
        for img2 in img_comparisons:
            if img1 != img2:

                img2_tensor = images[img2].to(device)

                with torch.no_grad():
                    # Make prediction
                    output = model(img1_tensor, img2_tensor).squeeze()
                    prediction = torch.sigmoid(output).item()

                if prediction > 0.5:
                    rankings[img1], rankings[img2] = elo_rating(rankings[img1], rankings[img2], 1)
                else:
                    rankings[img1], rankings[img2] = elo_rating(rankings[img1], rankings[img2], 0)


    sorted_rankings = sorted(rankings.items(), key=lambda item: item[1], reverse=True)
    print("Ranking completed.")
    for rank, (image_file, score) in enumerate(sorted_rankings, 1):
        print(f"Rank {rank}: {image_file} with score {score}")

    # Save rankings to a file
    rankings_path = os.path.join(working_dir, 'ranker', 'rankings.json')
    with open(rankings_path, 'w') as f:
        json.dump(sorted_rankings, f, indent=4)
    print(f"Rankings saved to {rankings_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank Images")
    parser.add_argument("-w", "--working_dir", type=str, required=True, help="Working Directory in ILM Format.")
    parser.add_argument("-c", "--comparisons", type=int, required=True, help="No. of comparissons per image.")
    
    args = parser.parse_args()
    rank(args.working_dir, args.comparisons)