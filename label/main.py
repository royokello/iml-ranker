import argparse
import json
import os
from flask import Flask, jsonify, render_template, request, send_from_directory


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