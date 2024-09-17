# dataset.py

import os
import json
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Updated label mapping to include "both" and "neither"
LABEL_MAP = {
    "left": 0,
    "right": 1,
    "both": 2,
    "neither": 3
}

class PreferenceDataset(Dataset):
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
        image_dir: str,
        labels: Dict[str, str],
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Initializes the PreferenceDataset.

        Args:
            image_dir (str): Directory containing all images. Filenames should correspond to image IDs (e.g., "363.png").
            labels (Dict[str, str]): Dictionary with keys as "id1_id2" and values as "left", "right", "both", or "neither".
            transform (Optional[transforms.Compose]): Transformations to apply to the images.
        """
        self.image_dir = image_dir
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
        id1, id2 = pair_key.split('_')

        # Construct image file paths
        img1_path = self._get_image_path(id1)
        img2_path = self._get_image_path(id2)

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

    def _get_image_path(self, image_id: str) -> str:
        """
        Constructs the full path to an image given its ID.

        Args:
            image_id (str): Image ID.

        Returns:
            str: Full path to the image file.
        """
        filename = f"{image_id}.png"  # Update extension if necessary (e.g., ".jpg")
        full_path = os.path.join(self.image_dir, filename)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Image file '{full_path}' not found.")
        return full_path

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


def load_labels(labels_file: str) -> Dict[str, str]:
    """
    Loads label data from a JSON file.

    Args:
        labels_file (str): Path to the JSON file containing labels.

    Returns:
        Dict[str, str]: Dictionary with keys as "id1_id2" and values as "left", "right", "both", or "neither".
    """
    if not os.path.isfile(labels_file):
        raise FileNotFoundError(f"Labels file '{labels_file}' not found.")

    with open(labels_file, 'r') as f:
        labels = json.load(f)

    # Validate labels
    for pair, label in labels.items():
        if label.lower() not in LABEL_MAP:
            raise ValueError(f"Invalid label '{label}' for pair '{pair}'. Must be one of {list(LABEL_MAP.keys())}.")

    return labels


def split_dataset(
    labels: Dict[str, str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        labels (Dict[str, str]): Entire label dictionary.
        train_ratio (float): Proportion of data to use for training.
        val_ratio (float): Proportion of data to use for validation.
        test_ratio (float): Proportion of data to use for testing.
        random_state (int): Seed for random number generator.

    Returns:
        Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]: Dictionaries for train, validation, and test sets.
    """
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.")

    pairs = list(labels.keys())
    train_val_pairs, test_pairs = train_test_split(
        pairs, test_size=test_ratio, random_state=random_state, shuffle=True
    )
    train_pairs, val_pairs = train_test_split(
        train_val_pairs,
        test_size=val_ratio / (train_ratio + val_ratio),
        random_state=random_state,
        shuffle=True,
    )

    train_labels = {pair: labels[pair] for pair in train_pairs}
    val_labels = {pair: labels[pair] for pair in val_pairs}
    test_labels = {pair: labels[pair] for pair in test_pairs}

    return train_labels, val_labels, test_labels


def get_data_transforms(
    image_size: Tuple[int, int] = (256, 256),
    augment: bool = True,
) -> transforms.Compose:
    """
    Defines the data transformations for training and validation.

    Args:
        image_size (Tuple[int, int]): Desired image size (height, width).
        augment (bool): Whether to include data augmentation.

    Returns:
        transforms.Compose: Composed transformations.
    """
    if augment:
        transform = transforms.Compose([
            transforms.Resize(image_size),  # Resize to 256x256
            transforms.RandomHorizontalFlip(),  # Data augmentation
            transforms.RandomRotation(15),      # Data augmentation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Data augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                                 std=[0.229, 0.224, 0.225]),  # ImageNet std
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),  # Resize to 256x256
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                                 std=[0.229, 0.224, 0.225]),  # ImageNet std
        ])
    return transform


def create_data_loaders(
    image_dir: str,
    labels_file: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (256, 256),
    train_ratio: float = 0.75,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates data loaders for training, validation, and testing.

    Args:
        image_dir (str): Directory containing all images.
        labels_file (str): Path to the JSON file containing labels.
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
    # Load labels
    labels = load_labels(labels_file)

    # Split dataset
    train_labels, val_labels, test_labels = split_dataset(
        labels, train_ratio, val_ratio, test_ratio, random_state
    )

    # Define transforms
    train_transform = get_data_transforms(image_size=image_size, augment=True)
    val_test_transform = get_data_transforms(image_size=image_size, augment=False)

    # Create dataset instances
    train_dataset = PreferenceDataset(
        image_dir=image_dir, labels=train_labels, transform=train_transform
    )
    val_dataset = PreferenceDataset(
        image_dir=image_dir, labels=val_labels, transform=val_test_transform
    )
    test_dataset = PreferenceDataset(
        image_dir=image_dir, labels=test_labels, transform=val_test_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader
