# dataset.py

import os
import json
import csv
import glob
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# Updated label mapping to include "both" and "neither"
LABEL_MAP = {
    "left": 0,
    "right": 1,
    "both": 2,
    "neither": 3
}

class IMLRankDataset(Dataset):
    """
    PyTorch Dataset for Pairwise Image Preference Learning.

    Each sample consists of two images and a label indicating preference:
    - Label 0: "left"
    - Label 1: "right"
    - Label 2: "both"
    - Label 3: "neither"
    """

    def __init__(
        self,
        root_dir: str,
        labels: Dict[str, str],
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Initializes the IMLRankDataset.

        Args:
            root_dir (str): Root directory containing album folders with src/ images.
            labels (Dict[str, str]): Dictionary with keys as "album/id1|||id2" and values as "left", "right", "both", or "neither".
            transform (Optional[transforms.Compose]): Transformations to apply to the images.
        """
        self.root_dir = root_dir
        self.labels = labels
        self.transform = transform
        self.pairs = list(labels.keys())

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves the pair of images and the corresponding label.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing:
                - Image 1 tensor
                - Image 2 tensor
                - Label tensor (0: "left", 1: "right", 2: "both", 3: "neither")
        """
        pair_key = self.pairs[idx]
        
        # Parse album and image IDs from the pair key
        if '/' in pair_key:
            album, pair = pair_key.split('/', 1)
            # Handle multiple underscores in image IDs
            parts = pair.split('|||')
            # The pattern should be id1|||id2, but ids themselves might contain underscores
            # We'll assume the last part separates the two IDs
            if len(parts) > 2:
                id2 = parts[-1]  # Last part is id2
                id1 = '|||'.join(parts[:-1])  # Everything else is id1
            else:
                id1, id2 = parts
        else:
            # Fallback for old format
            parts = pair_key.split('|||')
            # Handle multiple underscores in the image IDs
            if len(parts) > 2:
                id2 = parts[-1]  # Last part is id2
                id1 = '|||'.join(parts[:-1])  # Everything else is id1
            else:
                id1, id2 = parts
            album = None

        # Construct image file paths
        img1_path = self._get_image_path(album, id1)
        img2_path = self._get_image_path(album, id2)

        # Load images
        image1 = self._load_image(img1_path)
        image2 = self._load_image(img2_path)

        # Apply transformations if provided
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        # Determine label
        label_str = self.labels[pair_key].lower()
        if label_str in LABEL_MAP:
            label = torch.tensor(LABEL_MAP[label_str], dtype=torch.long)
        else:
            raise ValueError(f"Invalid label '{label_str}' for pair '{pair_key}'.")

        return image1, image2, label

    def _get_image_path(self, album: Optional[str], image_id: str) -> str:
        """
        Constructs the full path to an image given its album and ID.

        Args:
            album (Optional[str]): Album folder name, or None for root-level images.
            image_id (str): Image ID.

        Returns:
            str: Full path to the image file.
        """
        # Try multiple extensions
        extensions = ['.png', '.jpg', '.jpeg']
        
        if album:
            for ext in extensions:
                # Construct path directly in the album directory (no 'src' subdirectory)
                path = os.path.join(self.root_dir, f"{image_id}{ext}")
                if os.path.isfile(path):
                    return path
            
            # If not found, raise an error
            raise FileNotFoundError(f"Image file for ID '{image_id}' in album '{album}' not found in {self.root_dir}.")
        else:
            # Old format fallback
            for ext in extensions:
                path = os.path.join(self.root_dir, f"{image_id}{ext}")
                if os.path.isfile(path):
                    return path
                
            raise FileNotFoundError(f"Image file for ID '{image_id}' not found.")

    def _load_image(self, path: str) -> Image.Image:
        """
        Loads an image from the given path.

        Args:
            path (str): Path to the image file.

        Returns:
            Image.Image: Loaded PIL image.
        """
        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except Exception as e:
            raise IOError(f"Error loading image '{path}': {e}")


def load_csv_labels(csv_file: str, album: str) -> Dict[str, str]:
    """
    Loads label data from a CSV file.

    Args:
        csv_file (str): Path to the CSV file containing labels.
        album (str): Album name to prefix the image IDs.

    Returns:
        Dict[str, str]: Dictionary with keys as "album/id1|||id2" and values as "left", "right", "both", or "neither".
    """
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"Labels file '{csv_file}' not found.")

    labels = {}
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip header row
        
        # Determine column indexes based on header
        if header:
            try:
                img1_idx = header.index('img_1')
                img2_idx = header.index('img_2')
                choice_idx = header.index('rank')
            except ValueError:
                # Fallback to positional if header doesn't match expected format
                img1_idx, img2_idx, choice_idx = 0, 1, 2
        else:
            # No header, assume positional
            img1_idx, img2_idx, choice_idx = 0, 1, 2
        
        for row in reader:
            if len(row) > max(img1_idx, img2_idx, choice_idx):
                id1 = row[img1_idx].strip()
                id2 = row[img2_idx].strip()
                choice = row[choice_idx].strip().lower()
                
                # Validate choice
                if choice not in LABEL_MAP:
                    print(f"Warning: Invalid label '{choice}' in {csv_file}. Skipping.")
                    continue
                
                # Create key with album prefix and special delimiter
                key = f"{album}/{id1}|||{id2}"
                labels[key] = choice

    return labels


def split_dataset(
    labels: Dict[str, str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[Dict[str, str], Dict[str, str]]: # Return only train and val labels
    """
    Splits the dataset into training and validation sets.

    Args:
        labels (Dict[str, str]): Entire label dictionary.
        train_ratio (float): Proportion of data to use for training.
        val_ratio (float): Proportion of data to use for validation.
        random_state (int): Seed for random number generator.

    Returns:
        Tuple[Dict[str, str], Dict[str, str]]: Dictionaries for train and validation sets.
    """
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and abs(train_ratio + val_ratio - 1.0) < 1e-6):
        raise ValueError("Train and validation ratios must be between 0 and 1, and sum to 1.")

    import random
    random.seed(random_state)
    
    all_keys = list(labels.keys())
    random.shuffle(all_keys)
    
    # Calculate split indices
    num_samples = len(all_keys)
    train_end = int(train_ratio * num_samples)

    # Create splits
    train_keys = all_keys[:train_end]
    val_keys = all_keys[train_end:] # The rest goes to validation

    # Create label dictionaries for each split
    train_labels = {key: labels[key] for key in train_keys}
    val_labels = {key: labels[key] for key in val_keys}

    return train_labels, val_labels


def get_data_transforms(
    image_size: Tuple[int, int] = (224, 224),
    augment: bool = True,
) -> transforms.Compose:
    """
    Defines the data transformations for training and validation.

    Args:
        image_size (Tuple[int, int]): Desired image size (height, width).
        augment (bool): Whether to include data augmentation.

    Returns:
        transforms.Compose: Composition of transformations to apply.
    """
    if augment:
        transform = transforms.Compose([
            transforms.Resize((image_size[0] + 20, image_size[1] + 20)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    return transform


def create_data_loaders(
    root_dir: str,
    labels: Dict[str, str] = None,
    labels_csv: str = None,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    train_ratio: float = 0.8, # Default to 80/20 split
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader]: # Return only train and val loaders
    """
    Creates data loaders for training, validation, and testing.

    Args:
        root_dir (str): Root directory containing album folders.
        labels (Dict[str, str], optional): Preloaded labels dictionary. If None, labels will be loaded from CSV files.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.
        image_size (Tuple[int, int]): Desired image size.
        train_ratio (float): Proportion of data to use for training.
        val_ratio (float): Proportion of data to use for validation.
        test_ratio (float): Proportion of data to use for testing.
        random_state (int): Seed for random number generator.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Data loaders for train, validation, and test sets.
    """
    # If labels not provided, try loading from direct CSV path or from album folders
    if labels is None:
        labels = {}
        
        # If a specific labels CSV file is provided, use it
        if labels_csv and os.path.isfile(labels_csv):
            # Extract folder name from root_dir to use as the album name
            album_name = os.path.basename(os.path.normpath(root_dir))
            album_labels = load_csv_labels(labels_csv, album_name)
            labels.update(album_labels)
            print(f"Loaded {len(album_labels)} labels from {labels_csv}")
        else:
            # Fallback to searching for CSVs in album folders
            album_folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            
            if not album_folders:
                raise ValueError(f"No album folders found in {root_dir}")
            
            for album in album_folders:
                album_dir = os.path.join(root_dir, album)
                
                # Check if album has a src directory
                if not os.path.isdir(os.path.join(album_dir, 'src')):
                    continue
                    
                # Look for rank_labels.csv in the album directory
                csv_path = os.path.join(album_dir, 'rank_labels.csv')
                if os.path.isfile(csv_path):
                    album_labels = load_csv_labels(csv_path, album)
                    labels.update(album_labels)
                    print(f"Loaded {len(album_labels)} labels from {csv_path}")
                else:
                    print(f"No rank_labels.csv found in {album_dir}")
    
    if not labels:
        raise ValueError("No valid labels found in any album.")
    
    print(f"Total label pairs: {len(labels)}")

    # Split dataset
    train_labels, val_labels = split_dataset(
        labels, train_ratio, val_ratio, random_state
    )
    
    print(f"Split into {len(train_labels)} training and {len(val_labels)} validation pairs")

    # Define transforms
    train_transform = get_data_transforms(image_size=image_size, augment=True)
    val_test_transform = get_data_transforms(image_size=image_size, augment=False)

    # Create dataset instances
    train_dataset = IMLRankDataset(
        root_dir=root_dir, labels=train_labels, transform=train_transform
    )
    val_dataset = IMLRankDataset(
        root_dir=root_dir, labels=val_labels, transform=val_test_transform
    )
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader
