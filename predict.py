import torch
from PIL import Image
from torchvision import transforms

def predict(device: torch.device, model, image_1_path: str, image_2_path: str) -> bool:
    """
    """
    # Define the transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load and preprocess the images
    image1 = Image.open(image_1_path).convert("RGB")
    image2 = Image.open(image_2_path).convert("RGB")
    image1 = transform(image1).unsqueeze(0).to(device)
    image2 = transform(image2).unsqueeze(0).to(device)

    # Set the model to evaluation mode
    # model.eval()

    with torch.no_grad():
        # Make prediction
        output = model(image1, image2).squeeze()
        prediction = torch.sigmoid(output).item()

    # Return True if the first image is preferred (prediction > 0.5), False otherwise
    return prediction > 0.5