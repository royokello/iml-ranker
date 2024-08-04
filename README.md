# iml-ranker
IML-Ranker is a Python library for ranking images using machine learning and the Elo rating system. It collects user preferences via pairwise comparisons, extracts image features, and trains models to predict preferences, enabling efficient identification of top-rated images.


## Label

`python -m label.main -w "path to working directory"`

## Train

`python train.py -w "path to working directory" -e "epochs" -c "checkpoints" -b "base model name (optional)"`
