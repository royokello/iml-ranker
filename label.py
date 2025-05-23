#!/usr/bin/env python
import json
import os
import csv
import sys
import re
import argparse
from flask import Flask, jsonify, render_template_string, request, send_from_directory
from PIL import Image
import numpy as np

# HTML Template as a string
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IML Ranker - {{ album_name }}</title>
    <style>
        body {
            display: flex;
            flex-direction: column;
            align-items: center;
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #ffffff;
        }
        .header {
            color: #333333;
            width: 100%;
            padding: 20px 0 10px 0;
            text-align: center;
            border-bottom: 1px solid #eeeeee;
        }
        .header h1 {
            margin: 0;
            font-weight: bold;
        }
        .header .subtitle {
            color: #888888;
            margin: 4px 0 8px 0;
        }
        .header .counters {
            color: #666666;
            margin-top: 4px;
        }
        .main-container {
            display: flex;
            flex-direction: row;
            justify-content: center;
            align-items: flex-start;
            width: 100%;
            max-width: 1200px;
            padding: 20px;
            box-sizing: border-box;
        }
        .image-section {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0 20px;
        }
        .image-container {
            width: 384px;
            height: 384px;
            object-fit: contain;
            border: 2px solid #ccc;
            background-color: white;
        }
        .central-controls {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0 20px;
        }
        .switch-button, .random-pair-button {
            padding: 10px 20px;
            margin: 10px 0;
            font-size: 16px;
            cursor: pointer;
            border: none;
            border-radius: 5px;
            color: white;
            transition: background-color 0.3s ease;
        }
        .switch-button {
            background-color: #FFC107; /* Amber */
        }
        .switch-button:hover {
            background-color: #ffb300;
        }
        .random-pair-button {
            background-color: #009688; /* Teal */
        }
        .random-pair-button:hover {
            background-color: #00796b;
        }
        .labels-container {
            display: flex;
            flex-direction: column;
            align-items: stretch;
            width: 200px;
            margin-top: 20px;
        }
        .label-button {
            padding: 10px;
            margin: 5px 0;
            font-size: 16px;
            cursor: pointer;
            border: none;
            border-radius: 5px;
            color: white;
            transition: background-color 0.3s ease;
        }
        .label-left {
            background-color: #4CAF50; /* Green */
        }
        .label-left:hover {
            background-color: #43a047;
        }
        .label-right {
            background-color: #f44336; /* Red */
        }
        .label-right:hover {
            background-color: #d32f2f;
        }
        .label-both {
            background-color: #2196F3; /* Blue */
        }
        .label-both:hover {
            background-color: #1976d2;
        }
        .label-neither {
            background-color: #9E9E9E; /* Grey */
        }
        .label-neither:hover {
            background-color: #757575;
        }
        .labels-summary {
            width: 100%;
            margin-top: 20px;
            background-color: #ffffff;
            padding: 15px;
            border: 1px solid #ccc;
            border-radius: 8px;
        }
        .labels-summary h3 {
            margin-top: 0;
            text-align: center;
        }
        .stat-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }
        .stat-item:last-child {
            border-bottom: none;
        }
        .image-controls {
            display: flex;
            justify-content: center;
            width: 100%;
            margin: 10px 0;
        }
        .randomize-button {
            padding: 8px 15px;
            margin: 0 5px;
            font-size: 14px;
            cursor: pointer;
            border: none;
            border-radius: 5px;
            color: white;
            background-color: #673AB7; /* Deep Purple */
            transition: background-color 0.3s ease;
        }
        .randomize-button:hover {
            background-color: #5E35B1;
        }
        @media (max-width: 992px) {
            .main-container {
                flex-direction: column;
                align-items: center;
            }
            .central-controls {
                margin: 20px 0;
            }
            .labels-container {
                width: 250px;
            }
        }
        @media (max-width: 600px) {
            .image-container {
                width: 280px;
                height: 280px;
            }
            .labels-container {
                width: 100%;
            }
            .random-pair-button, .switch-button {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>IML Rank</h1>
        <div class="subtitle">royokello</div>
        <div class="counters">Total Images: {{ total_images }} | Total Labels: {{ total_labels }}</div>
    </div>
    
    <div class="main-container">
        <div class="image-section">
            <h2>Image 1</h2>
            <div class="image-controls">
                <button class="randomize-button" onclick="randomizeImage(1)">Randomize Image 1</button>
            </div>
            <img id="image-1" class="image-container" src="" alt="Image 1">
        </div>
        
        <div class="central-controls">
            <button id="random-pair-button" class="random-pair-button" onclick="getRandomPair()">Get Random Pair</button>
            <button id="switch-button" class="switch-button" onclick="switchImages()">Switch Images</button>
            
            <div class="labels-container">
                <h3>Select Label</h3>
                <button class="label-button label-left" onclick="addLabel('left')">Left is Better</button>
                <button class="label-button label-right" onclick="addLabel('right')">Right is Better</button>
                <button class="label-button label-both" onclick="addLabel('both')">Both are Good</button>
                <button class="label-button label-neither" onclick="addLabel('neither')">Neither is Good</button>
            </div>
            
            <div class="labels-summary">
                <h3>Labels Summary</h3>
                <div class="stat-item">
                    <span>Left is Better:</span>
                    <span id="left-count">{{ label_stats.left }}</span>
                </div>
                <div class="stat-item">
                    <span>Right is Better:</span>
                    <span id="right-count">{{ label_stats.right }}</span>
                </div>
                <div class="stat-item">
                    <span>Both are Good:</span>
                    <span id="both-count">{{ label_stats.both }}</span>
                </div>
                <div class="stat-item">
                    <span>Neither is Good:</span>
                    <span id="neither-count">{{ label_stats.neither }}</span>
                </div>
            </div>
        </div>
        
        <div class="image-section">
            <h2>Image 2</h2>
            <div class="image-controls">
                <button class="randomize-button" onclick="randomizeImage(2)">Randomize Image 2</button>
            </div>
            <img id="image-2" class="image-container" src="" alt="Image 2">
        </div>
    </div>
    
    <script>
        // Initialize variables with server-side data
        const total_images = parseInt("{{ total_images }}");
        let total_labels = parseInt("{{ total_labels }}");
        let label_stats = {
            'left': parseInt("{{ label_stats.left }}"),
            'right': parseInt("{{ label_stats.right }}"),
            'both': parseInt("{{ label_stats.both }}"),
            'neither': parseInt("{{ label_stats.neither }}")
        };
        
        // Get the counter element in the header
        const totalLabelsCounter = document.querySelector('.counters');
        
        // Variables to track current image identifiers
        let img_1 = "";
        let img_2 = "";
        
        // DOM elements
        const img1Element = document.getElementById('image-1');
        const img2Element = document.getElementById('image-2');
        
        // Label count elements
        const leftCountElement = document.getElementById('left-count');
        const rightCountElement = document.getElementById('right-count');
        const bothCountElement = document.getElementById('both-count');
        const neitherCountElement = document.getElementById('neither-count');
        
        // Initialize with random pair
        getRandomPair();
        
        /**
         * Fetches a random pair of images.
         */
        function getRandomPair() {
            fetch('/random')
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    img_1 = data.img_1;
                    img_2 = data.img_2;
                    updateImages(img_1, img_2);
                })
                .catch(error => {
                    console.error('Error fetching random pair:', error);
                    alert('Error loading random pair. Please ensure there are at least 2 images.');
                });
        }
        
        /**
         * Updates the displayed images.
         */
        function updateImages(img1Id, img2Id) {
            img1Element.src = `/image/${encodeURIComponent(img1Id)}`;
            img2Element.src = `/image/${encodeURIComponent(img2Id)}`;
        }
        
        /**
         * Switches the positions of the current images.
         */
        function switchImages() {
            if (img_1 === "" || img_2 === "") return;
            
            fetch('/switch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    img_1: img_1,
                    img_2: img_2
                })
            })
            .then(response => response.json())
            .then(data => {
                img_1 = data.img_1;
                img_2 = data.img_2;
                updateImages(data.img_1, data.img_2);
            })
            .catch(error => {
                console.error('Error switching images:', error);
            });
        }
        
        /**
         * Adds a label to the current image pair.
         * @param {string} choice - The label choice ('left', 'right', 'both', 'neither').
         */
        function addLabel(choice) {
            // Check if we have valid image IDs
            if (!img_1 || !img_2 || img_1 === "" || img_2 === "") {
                console.error('Cannot add label: Invalid image IDs');
                alert('Error: Cannot add label with invalid images. Please get a new random pair.');
                return;
            }
            
            console.log(`Submitting label: img_1=${img_1}, img_2=${img_2}, choice=${choice}`);
            
            fetch('/label', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    'img_1': img_1,
                    'img_2': img_2,
                    'choice': choice,
                })
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(errorData => {
                        throw new Error(`Server error (${response.status}): ${errorData.error || 'Unknown error'}`);
                    });
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    console.log('Label added successfully');
                    // Update total labels
                    total_labels++;
                    
                    // Update label stats
                    label_stats[choice]++;
                    updateLabelStats();
                    
                    // Get new random pair
                    getRandomPair();
                } else if (data.error) {
                    console.error(`Server reported error: ${data.error}`);
                    alert(`Error: ${data.error}`);
                }
            })
            .catch(error => {
                console.error('Error adding label:', error);
                alert(`Failed to add label: ${error.message}`);
            });
        }
        
        /**
         * Updates the label statistics display.
         */
        function updateLabelStats() {
            leftCountElement.textContent = label_stats.left;
            rightCountElement.textContent = label_stats.right;
            bothCountElement.textContent = label_stats.both;
            neitherCountElement.textContent = label_stats.neither;
            
            // Update the total labels counter in the header
            totalLabelsCounter.textContent = `Total Images: ${total_images} | Total Labels: ${total_labels}`;
        }
        
        /**
         * Randomizes a single image (either 1 or 2).
         * @param {number} imageNumber - The image number to randomize (1 or 2).
         */
        function randomizeImage(imageNumber) {
            if (imageNumber !== 1 && imageNumber !== 2) return;
            
            // Keep track of the current images
            const currentImg1 = img_1;
            const currentImg2 = img_2;
            
            fetch('/random_single')
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    if (imageNumber === 1) {
                        img_1 = data.img;
                        updateImages(img_1, img_2);
                    } else {
                        img_2 = data.img;
                        updateImages(img_1, img_2);
                    }
                })
                .catch(error => {
                    console.error(`Error randomizing image ${imageNumber}:`, error);
                    alert('Error loading random image. Please ensure there are enough images.');
                });
        }
    </script>
</body>
</html>
"""

# Flask application
app = Flask(__name__)

# Global variables
album_dir = ""
src_dir = ""
labels_path = ""
labels = {}
image_ids = []

@app.route('/')
def index():
    global src_dir, labels

    # Calculate the total number of image files in the src directory
    total_images = sum(1 for entry in os.scandir(src_dir) if entry.is_file() and
                       entry.name.lower().endswith(('.png', '.jpg', '.jpeg')))

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

    # Get album name from directory path
    album_name = os.path.basename(album_dir)

    # Render the index.html template with the updated data
    return render_template_string(
        HTML_TEMPLATE,
        total_images=total_images,
        total_labels=total_labels,
        label_stats=label_stats,
        album_name=album_name,
    )

@app.route('/image/<path:img_id>')
def get_image(img_id):
    global src_dir
    # Find the file with the given id regardless of extension
    for ext in ['.png', '.jpg', '.jpeg']:
        filename = f"{img_id}{ext}"
        if os.path.exists(os.path.join(src_dir, filename)):
            return send_from_directory(src_dir, filename)
    
    # If no matching file was found
    return "Image not found", 404

@app.route('/label', methods=['POST'])
def label_image():
    global labels, labels_path
    try:
        print("Received label request")
        data = request.json
        print(f"Request data: {data}")
        
        if not data:
            print("Error: No data provided")
            return jsonify(success=False, error="No data provided"), 400
            
        # Validate required fields
        if 'img_1' not in data or 'img_2' not in data or 'choice' not in data:
            missing = []
            if 'img_1' not in data: missing.append('img_1')
            if 'img_2' not in data: missing.append('img_2')
            if 'choice' not in data: missing.append('choice')
            error_msg = f"Missing required fields: {', '.join(missing)}"
            print(f"Error: {error_msg}")
            return jsonify(success=False, error=error_msg), 400
            
        img_1_id = data['img_1']
        img_2_id = data['img_2']
        choice = data['choice']
        
        # Validate choice
        valid_choices = ['left', 'right', 'both', 'neither']
        if choice.lower() not in valid_choices:
            error_msg = f"Invalid choice '{choice}'. Must be one of: {', '.join(valid_choices)}"
            print(f"Error: {error_msg}")
            return jsonify(success=False, error=error_msg), 400
        
        # Validate that the image IDs exist in our image_ids list
        global image_ids
        if img_1_id not in image_ids:
            error_msg = f"Image ID '{img_1_id}' not found in available images"
            print(f"Error: {error_msg}")
            return jsonify(success=False, error=error_msg), 400
            
        if img_2_id not in image_ids:
            error_msg = f"Image ID '{img_2_id}' not found in available images"
            print(f"Error: {error_msg}")
            return jsonify(success=False, error=error_msg), 400
        
        # Create a consistent pair ID for storage
        # Sort the IDs to ensure consistent ordering
        sorted_ids = sorted([img_1_id, img_2_id])
        pair_id = f"{sorted_ids[0]}|||{sorted_ids[1]}"
        
        # Store in our labels dictionary
        labels[pair_id] = choice
        
        # Save to CSV
        with open(labels_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['img_1', 'img_2', 'rank'])
            for pair, label_choice in labels.items():
                img1, img2 = pair.split('|||')
                writer.writerow([img1, img2, label_choice])
        
        print(f"Label saved successfully: {img_1_id}, {img_2_id}, {choice}")
        return jsonify(success=True)
    except Exception as e:
        # Log the error
        print(f"Error in label_image: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return a proper JSON error response
        return jsonify(success=False, error=str(e)), 500

@app.route('/random', methods=['GET'])
def get_random():
    global image_ids
    import random
    if len(image_ids) <= 1:
        return jsonify({
            "error": "Not enough images",
            "total": len(image_ids)
        }), 400
        
    # Select two random, distinct image IDs (using the actual filenames)
    img_1, img_2 = random.sample(image_ids, 2)
    
    return jsonify({
        "img_1": img_1,
        "img_2": img_2
    })

@app.route('/random_single', methods=['GET'])
def get_random_single():
    global image_ids
    import random
    if len(image_ids) <= 0:
        return jsonify({
            "error": "No images available",
            "total": len(image_ids)
        }), 400
        
    # Select one random image ID (using the actual filename)
    img = random.choice(image_ids)
    
    return jsonify({
        "img": img
    })

@app.route('/switch', methods=['POST'])
def switch_images():
    data = request.json
    img_1 = data.get('img_1')
    img_2 = data.get('img_2')
    
    global image_ids
    if img_1 is None or img_2 is None or img_1 not in image_ids or img_2 not in image_ids:
        return jsonify({"error": "Invalid image identifiers"}), 400
    
    return jsonify({
        "img_1": img_2,  # Switched!
        "img_2": img_1   # Switched!
    })

def load_labels_from_csv(csv_path):
    """Load labels from a CSV file into a dictionary."""
    labels = {}
    if not os.path.exists(csv_path):
        # Create an empty CSV with headers
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['img_1', 'img_2', 'rank'])
        return labels
    
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        # Skip header row
        header = next(reader, None)
        
        if header:
            # Find column indices
            try:
                img1_idx = header.index('img_1')
                img2_idx = header.index('img_2')
                choice_idx = header.index('rank')
            except ValueError:
                # Fallback to positional if header doesn't have expected columns
                img1_idx, img2_idx, choice_idx = 0, 1, 2
            
            for row in reader:
                if len(row) > max(img1_idx, img2_idx, choice_idx):
                    img1 = row[img1_idx].strip()
                    img2 = row[img2_idx].strip()
                    choice = row[choice_idx].strip()
                    # Create a consistent pair ID using the same format as when saving
                    sorted_ids = sorted([img1, img2])
                    pair_id = f"{sorted_ids[0]}|||{sorted_ids[1]}"
                    labels[pair_id] = choice
    
    return labels

def get_image_ids(directory):
    """Get list of image IDs from filenames in a directory."""
    image_ids = []
    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)):
            name, ext = os.path.splitext(f)
            if ext.lower() in ['.jpg', '.jpeg', '.png']:
                image_ids.append(name)
    return image_ids

def start_labeler(project_dir, stage):
    global album_dir, src_dir, labels_path, labels, image_ids
    
    # Set album directory to the project directory
    album_dir = project_dir
    
    # Check project directory
    if not os.path.isdir(album_dir):
        print(f"Project directory not found: {album_dir}")
        print("Creating the project directory...")
        os.makedirs(album_dir, exist_ok=True)
    
    # Set up stage directory
    stage_dir = f"stage_{stage}"
    src_dir = os.path.join(album_dir, stage_dir)
    
    # Create stage directory if it doesn't exist
    if not os.path.isdir(src_dir):
        print(f"Stage directory not found: {src_dir}")
        print("Creating the stage directory...")
        os.makedirs(src_dir, exist_ok=True)
    
    # Set up labels path with stage-specific filename
    labels_path = os.path.join(album_dir, f"{stage_dir}_rank_labels.csv")
    labels = load_labels_from_csv(labels_path)
    
    # Get image IDs
    image_ids = get_image_ids(src_dir)
    print(f"Found {len(image_ids)} images in {src_dir}")
    
    # Print start message
    project_name = os.path.basename(album_dir)
    print(f"Starting labeler for project: {project_name}, stage: {stage}")

def get_latest_stage(project_dir):
    """Find the latest stage directory in the project directory."""
    stage_pattern = re.compile(r'stage_(\d+)')
    stages = []
    
    # Find all directories that match the stage pattern
    for item in os.listdir(project_dir):
        item_path = os.path.join(project_dir, item)
        if os.path.isdir(item_path):
            match = stage_pattern.match(item)
            if match:
                stage_num = int(match.group(1))
                stages.append((stage_num, item))
    
    if not stages:
        print(f"No stage directories found in {project_dir}")
        return None
    
    # Sort by stage number (descending) and return the highest
    stages.sort(reverse=True)
    return stages[0][1]

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start the IML Ranker labeling tool.')
    parser.add_argument('--project', required=True, help='Project directory path where files are kept')
    parser.add_argument('--stage', type=str, help='Stage number (optional, will use latest stage if not provided)')
    
    args = parser.parse_args()
    
    # If stage is not provided, find the latest stage directory
    if args.stage is None:
        stage_dir = get_latest_stage(args.project)
        if stage_dir is None:
            print("Error: No stage directories found in the project directory.")
            sys.exit(1)
        args.stage = stage_dir.replace('stage_', '')  # Extract just the number
    
    start_labeler(args.project, args.stage)
    app.run(debug=True)
