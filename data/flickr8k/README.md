# Flickr8k Dataset Setup

## Current Status
This directory contains a small synthetic dataset for demonstration purposes.

## Using Real Flickr8k Dataset

To use the real Flickr8k dataset:

1. Download from Kaggle:
   - Visit: https://www.kaggle.com/datasets/adityajn105/flickr8k
   - Download and extract to this directory

2. Expected structure:
   ```
   data/flickr8k/
   ├── images/           # 8091 images
   ├── captions.txt      # Image-caption pairs
   └── ...
   ```

3. Run preprocessing:
   ```bash
   python prepare_flickr8k.py --process-real
   ```

## Current Demo Dataset
- 3 sample images
- 5 captions per image
- Used for testing the pipeline
