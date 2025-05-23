import logging
import os
import time

import torch

from model import IMLRankModel

def log_print(message):
    print(message)
    logging.info(message)

def get_model_by_name(device: torch.device, directory: str, name: str) -> IMLRankModel | None:
    """
    Load a model by its name from the specified directory and move it to the specified device.
    
    Args:
        device (torch.device): Device to load the model on.
        directory (str): Directory containing the model files.
        name (str): Name prefix of the model file to load.
        
    Returns:
        IMLRankModel | None: Loaded model or None if no matching model is found.
    """
    
    model = IMLRankModel()  # Initialize ViT model architecture

    for file in os.listdir(directory):
        if file.startswith(name):
            model_path = os.path.join(directory, file)
            break
    else:
        return None

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    model = model.to(device)
    
    return model

def get_model_by_latest(device: torch.device, directory: str|None=None) -> IMLRankModel | None:
    """
    Load a model whose model name is the latest time from the specified directory and move it to the specified device.
    Priority is given to 'rank_model.pth' if it exists.
    
    Args:
        device (torch.device): Device to load the model on.
        directory (str | None): Directory containing the model files. If None, returns a new model.
        
    Returns:
        IMLRankModel | None: Loaded model or None if no models are found in the directory.
    """
    model = IMLRankModel()

    if directory and os.path.exists(directory):
        # First check if rank_model.pth exists
        rank_model_path = os.path.join(directory, 'rank_model.pth')
        if os.path.exists(rank_model_path):
            print(f"Found rank_model.pth, loading this model")
            model.load_state_dict(torch.load(rank_model_path, map_location=device))
            model = model.to(device)
            return model
            
        # If not, look for other model files
        model_files = [f for f in os.listdir(directory) if f.endswith('.pth')]
        if not model_files:
            return None

        latest_model = max(model_files)
        print(f"rank_model.pth not found, using latest model: {latest_model}")
        
        model_path = os.path.join(directory, latest_model)
        model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)
    
    return model

def generate_model_name(base_model: str | None, samples: int, epochs: int) -> str:
    """
    Generate a unique model name based on current timestamp, base model (if any), number of samples, and epochs.
    """
    result = f"{int(time.time())}"
    if base_model:
        result += f"_b={base_model}"
    
    result += f"_s={samples}_e={epochs}"
    
    return result