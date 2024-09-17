import argparse
import json
import os
from flask import Flask, jsonify, render_template, request, send_from_directory
import torch
from PIL import Image
from model import CustomResNet
from utils import get_model_by_latest, get_model_by_name
from torchvision import transforms
import utils
import numpy as np


app = Flask(__name__)

images_path = ""
labels_path = ""
low_res_path = ""
labels: dict[str, str] = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomResNet()
model_dir = ""

@app.route('/')
def index():
    global images_path, labels, model_dir

    # Calculate the total number of image files in the images directory
    total_images = sum(1 for entry in os.scandir(images_path) if entry.is_file())

    # Initialize total_labels and label_stats with all four labels
    total_labels = 0
    label_stats = {
        'left': 0,
        'right': 0,
        'both': 0,
        'neither': 0
    }

    # Iterate through all labels and update the counts
    for label in labels.values():
        normalized_label = label.lower()  # Ensure consistency in label format
        if normalized_label in label_stats:
            label_stats[normalized_label] += 1
            total_labels += 1
        else:
            # Log a warning if an unexpected label is encountered
            print(f"Warning: Encountered unexpected label '{label}'.")

    # Retrieve all model filenames without their extensions from the models directory
    models = [os.path.splitext(f)[0] for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]

    # Render the index.html template with the updated data
    return render_template(
        'index.html',
        total_images=total_images,
        total_labels=total_labels,
        label_stats=label_stats,
        models=models,
    )

@app.route('/image/<int:img_id>')
def get_image(img_id):
    global images_path
    return send_from_directory(images_path, f"{img_id}.png")

@app.route('/label', methods=['POST'])
def label_image():
    global labels, labels_path
    data = request.json
    if data:
        labels[data['id']] = data['choice']
        with open(labels_path, 'w') as f:
            json.dump(labels, f, indent=4)
        return jsonify(success=True)
    else:
        return jsonify(success=False)
    
@app.route('/predict', methods=['POST'])
def predict_choice():
    global low_res_path, device, model
    
    data = request.json
    if data:
        print(f"Predicting choice for {data['img_1_id']} vs {data['img_2_id']}...")

        image_1_path = os.path.join(low_res_path, f"{data['img_1_id']}.png")
        image_2_path = os.path.join(low_res_path, f"{data['img_2_id']}.png")

        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Adjust based on your model's requirements
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Use the same normalization as training
                                 std=[0.229, 0.224, 0.225]),
        ])

        try:
            image1 = Image.open(image_1_path).convert("RGB")
            image2 = Image.open(image_2_path).convert("RGB")
            image1 = transform(image1).unsqueeze(0).to(device)  # Shape: (1, 3, 256, 256)
            image2 = transform(image2).unsqueeze(0).to(device)  # Shape: (1, 3, 256, 256)
        except Exception as e:
            print(f"Error loading images: {e}")
            return jsonify(success=False, message="Error loading images."), 500

        with torch.no_grad():
            try:
                output = model(image1, image2)  # Expected shape: (1, 4)
                probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]  # Shape: (4,)
                predicted_idx = np.argmax(probabilities)
                classes = ['left', 'right', 'both', 'neither']
                prediction = classes[predicted_idx]
                
                # Round probabilities for readability
                probabilities = {cls: float(round(prob, 4)) for cls, prob in zip(classes, probabilities)}
                
                print(f"Result: {prediction}, Probabilities: {probabilities}")
                
                # Return the prediction and probabilities
                return jsonify({
                    "prediction": prediction,
                    "probabilities": probabilities
                }), 200
            except Exception as e:
                print(f"Error during prediction: {e}")
                return jsonify(success=False, message="Error during prediction."), 500
    else:
        return jsonify(success=False, message="Invalid data."), 400
    
def label(working_dir: str):
    global images_path, labels_path, labels, device, model, low_res_path, model_dir
    images_path = os.path.join(working_dir, 'cropper', 'output', '512p')
    labels_path = os.path.join(working_dir, 'ranker', 'labels.json')
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            labels = json.load(f)
    else:
        labels = {}
    model_dir = os.path.join(working_dir, 'ranker', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model = get_model_by_latest(device=device, directory=model_dir)
    if model:
        model.eval()
    low_res_path = os.path.join(working_dir, 'cropper', 'output', '256p')
    app.run(debug=True)

@app.route('/load_model/<string:model_name>')
def load_model(model_name):
    global device, model, model_dir
    try:
        model = get_model_by_name(device=device, name=model_name, directory=model_dir)
        model.eval()
        return jsonify(message=""), 200
    
    except Exception as e:
        return jsonify(message=str(e)), 400

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IML Ranker Labeler")
    parser.add_argument("-w", "--working_dir", type=str, required=True, help="Working Directory in ILM Format.")
    
    args = parser.parse_args()
    label(args.working_dir)