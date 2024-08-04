import logging
import os
import time

import torch

from model import ComparisonNet


def setup_logging(working_dir):
    log_file_path = os.path.join(working_dir, 'training.log')

    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def log_print(message):
    print(message)
    logging.info(message)

def get_model_by_name(device: torch.device, directory: str, name: str) -> ComparisonNet:
    """
    
    """
    
    model = ComparisonNet()  # Initialize your model architecture

    for file in os.listdir(directory):
        if file.startswith(name):
            model_path = os.path.join(directory, file)
            break
    else:
        raise ValueError(f"No model starting with {name} found in {directory}")

    model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)
    
    return model

def get_model_by_latest(device: torch.device, directory: str|None=None) -> ComparisonNet:
    """
    Load a model whose model name is the latest time from the specified directory and move it to the specified device.
    """
    model = ComparisonNet()

    if directory and os.path.exists(directory):
        model_files = [f for f in os.listdir(directory) if f.endswith('.pth')]
        if not model_files:
            raise ValueError(f"No model files found in {directory}")

        latest_model = max(model_files, key=lambda x: int(x.split('_')[0]))
        print(f"latest model: {latest_model}")
        
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