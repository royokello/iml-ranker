import argparse
import json
import os
from flask import Flask, render_template


app = Flask(__name__)
images_path = ""
labels_path = ""
labels: dict[str, bool] = {}

@app.route('/')
def index():
    global images_path, labels
    total_images = sum(1 for entry in os.scandir(images_path) if entry.is_file())
    total_labels = len(labels)
    return render_template('index.html', total_images=total_images, total_labels=total_labels)

def label(working_dir: str):
    global images_path, labels_path, labels
    images_path = os.path.join(working_dir, 'cropper', 'output', '512p')
    labels_path = os.path.join(working_dir, 'ranker', 'labels.json')
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            labels = json.load(f)
    else:
        labels = {}
    app.run(debug=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IML Ranker Labeler")
    parser.add_argument("-w", "--working_dir", type=str, required=True, help="Working Directory in ILM Format.")
    
    args = parser.parse_args()
    label(args.working_dir)