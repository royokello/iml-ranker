# iml-ranker

IML-Ranker is a Python library for ranking images using Vision Transformer (ViT) machine learning and the Elo rating system. It collects user preferences via pairwise comparisons, extracts image features, and trains models to predict preferences, enabling efficient identification of top-rated images.

## Directory Structure

The system now uses an album-based directory structure:

```
root_directory/
├── album1/
│   ├── src/                # Original source images
│   ├── src_culled/         # Selected/kept images (populated after culling)
│   ├── src_cropped/        # Cropped images used for ranking
│   ├── src_ranked/         # Extracted top-ranked images
│   ├── rankings.json       # Ranking results
│   └── cull_labels.csv     # Labeling data in CSV format
├── album2/
│   ├── ...
├── ...
├── models/                 # Trained models directory
└── rank_model.pth          # Main trained model
```

Each `cull_labels.csv` should have columns for image pairs and choices:
- `image1`: First image filename/ID
- `image2`: Second image filename/ID
- `choice`: One of "left", "right", "both", or "neither"

## Usage

### Labeling Images
```bash
python label.py ALBUM_DIR
```

The command takes the following positional argument:
- `ALBUM_DIR`: Directory containing the album images to be labeled.

This will start a web server to label images. Open a web browser and navigate to http://localhost:5000. A GUI will be displayed, allowing you to:
1. Compare pairs of images
2. Choose which image you prefer
3. Label as "both good" or "neither good"

The module will look for images in the `ALBUM_DIR/src_culled` directory and save the labels in `ALBUM_DIR/rank_labels.csv`.

## Train

Train the model using all labeled data across multiple albums:

```
python train.py -r "path to root directory" -e "epochs (default: 256)"
```

The trained model will be saved as `rank_model.pth` in the root directory.

## Rank

Rank images within one or all albums:

```
python rank.py -r "path to root directory" -a "album name (optional)" -c "comparisons (default: 10)"
```

If the `-a` parameter is omitted, all albums in the root directory will be processed.

The ranking process will:
1. Load the trained model (looking first in root directory, then in models subdirectory)
2. Use images from the `src_cropped` directory for each album
3. Save the ranking results as `rankings.json` in each album directory

## Extract

Extract the top-ranked images from a specific album:

```
python extract.py -r "path to root directory" -a "album name" -c "count of images to extract"
```

This will:
1. Read the rankings from `rankings.json` in the album directory
2. Copy the top-ranked images from the `src_cropped` directory 
3. Save them to the `src_ranked` directory within the album folder
