# iml-ranker

IML-Ranker is a Python application for ranking images using Vision Transformer (ViT) machine learning and the Elo rating system. It facilitates an iterative process of labeling, training, ranking, and extracting images through different stages.

## Project and Stage-Based Workflow

The system operates on a project directory, with data and models organized into stages. Each stage typically represents a step in the image selection and refinement process.

**Directory Structure:**

A typical project directory (`<project_dir>`) might look like this:

```
<project_dir>/
├── stage_0/                  # Initial set of images for labeling/ranking
│   ├── image1.jpg
│   └── image2.png
├── stage_0_rank_labels.csv   # Labels generated for images in stage_0
├── stage_0_rank_model.pth    # Model trained using labels from stage_0
├── stage_0_rankings.csv      # Ranking scores for images in stage_0
├── stage_1/                  # Images for the next stage (e.g., extracted from stage_0)
│   ├── 001_image_ranked_best.jpg
│   └── extraction_info.txt   # Information about the extraction process into this stage
├── stage_1_rank_labels.csv   # Labels for stage_1 images
├── stage_1_rank_model.pth    # Model trained on stage_1 data
├── stage_1_rankings.csv      # Rankings for stage_1 images
└── ...                       # Further stages
```

- **`stage_X/`**: Contains the source images for stage `X`.
- **`stage_X_rank_labels.csv`**: CSV file storing pairwise comparison labels for images in `stage_X`. Columns are typically `img_1_id,img_2_id,rank` (where rank can be `left`, `right`, `both`, `neither`).
- **`stage_X_rank_model.pth`**: The PyTorch model file trained using data from `stage_X`.
- **`stage_X_rankings.csv`**: CSV file storing the Elo ranking scores for images in `stage_X`. Columns are `image,score`.
- **`extraction_info.txt`**: Found in a `stage_Y` directory if it was created by the `extract.py` script from `stage_X` (where Y = X+1). Contains details about the extracted images.

## Usage

All scripts are run from the command line using Python.

### 1. Labeling Images (`label.py`)

Starts a web server for pairwise image labeling.

**Command:**
```bash
python label.py --project <project_dir> [--stage <stage_number>]
```

**Arguments:**
- `--project <project_dir>`: (Required) Path to the project directory.
- `--stage <stage_number>`: (Optional) The stage number (integer) whose images you want to label. Images will be sourced from `<project_dir>/stage_<stage_number>/`. If not provided, the script attempts to use the latest existing stage (highest `X` in `stage_X`). Labels are saved to `<project_dir>/stage_<stage_number>_rank_labels.csv`.

**Functionality:**
- Launches a Flask web application (default: `http://localhost:5000`).
- Displays pairs of images from the specified stage directory.
- Allows users to choose "left is better", "right is better", "both are good", or "neither is good".
- Saves labeling choices to the corresponding `_rank_labels.csv` file.

### 2. Training a Model (`train.py`)

Trains a Vision Transformer model based on the collected labels.

**Command:**
```bash
python train.py --project <project_dir> [--stage <stage_number>] [-e <epochs>]
```

**Arguments:**
- `--project <project_dir>`: (Required) Path to the project directory.
- `--stage <stage_number>`: (Optional) The stage number (integer) whose labels and images will be used for training. It reads labels from `<project_dir>/stage_<stage_number>_rank_labels.csv` and images from `<project_dir>/stage_<stage_number>/`. If not provided, the script uses the latest existing stage.
- `-e <epochs>`, `--epochs <epochs>`: (Optional) Number of training epochs. Default is 256.

**Functionality:**
- Loads image data and corresponding labels for the specified stage.
- Trains an `IMLRankModel`.
- Saves the trained model to `<project_dir>/stage_<stage_number>_rank_model.pth`.
- Logs training progress to `<project_dir>/stage_<stage_number>_rank_log.csv`.

### 3. Ranking Images (`rank.py`)

Uses a trained model to rank images within a specific stage using the Elo rating system.

**Command:**
```bash
python rank.py --project <project_dir> [--stage <stage_number>] [--comparisons <num>] [--batch_size <size>] [--verbose]
```

**Arguments:**
- `--project <project_dir>`: (Required) Path to the project directory.
- `--stage <stage_number>`: (Optional) The stage number (integer) whose images will be ranked. It loads the model `<project_dir>/stage_<stage_number>_rank_model.pth` and uses images from `<project_dir>/stage_<stage_number>/`. If not provided, the script uses the latest existing stage.
- `--comparisons <num>`: (Optional) Number of pairwise comparisons to simulate per image for Elo rating. Default is 8.
- `--batch_size <size>`: (Optional) Batch size for model inference during comparisons. Default is 8.
- `--verbose`: (Optional) Enable verbose logging output.

**Functionality:**
- Loads the pre-trained model for the specified stage.
- Loads images from the specified stage directory.
- Simulates pairwise comparisons using the model's predictions.
- Calculates Elo ratings for each image.
- Saves the final rankings to `<project_dir>/stage_<stage_number>_rankings.csv`.

### 4. Extracting Top Images (`extract.py`)

Copies the top-ranked images from a given stage to a new stage directory.

**Command:**
```bash
python extract.py --project <project_dir> --count <num_images> [--stage <stage_number>]
```

**Arguments:**
- `--project <project_dir>`: (Required) Path to the project directory.
- `--count <num_images>`: (Required) Number of top-ranked images to extract.
- `--stage <stage_number>`: (Optional) The stage number (integer) from which to extract images. It reads rankings from `<project_dir>/stage_<stage_number>_rankings.csv` and images from `<project_dir>/stage_<stage_number>/`. If not provided, the script uses the latest existing stage.

**Functionality:**
- Reads the rankings from the specified stage's `_rankings.csv` file.
- Copies the top `<num_images>` images from the source stage directory (`<project_dir>/stage_<stage_number>/`) to a new directory: `<project_dir>/stage_<stage_number + 1>/`.
- The output directory is cleared if it already exists.
- Creates an `extraction_info.txt` file in the new stage directory detailing the extracted images and source.
