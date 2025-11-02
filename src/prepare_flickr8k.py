"""
Prepare Flickr8k dataset - Create synthetic dataset for demonstration
Since Flickr8k requires manual download, we'll create a small synthetic dataset
and provide instructions for using real data
"""
import json
import os
from pathlib import Path
import random

def create_synthetic_dataset():
    """Create a small synthetic dataset for demonstration"""
    data_dir = Path("../data/flickr8k")
    data_dir.mkdir(exist_ok=True, parents=True)
    
    images_dir = data_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Sample captions for demonstration
    synthetic_data = {
        "sample_1.jpg": [
            "a dog running on the grass",
            "a brown dog playing outside",
            "a pet dog in the park",
            "a happy dog running in a field",
            "a dog enjoying outdoor activities"
        ],
        "sample_2.jpg": [
            "a mountain landscape with snow",
            "snow covered mountains under blue sky",
            "a beautiful mountain range",
            "mountains with white peaks",
            "scenic mountain view"
        ],
        "sample_3.jpg": [
            "a beach with waves",
            "ocean waves on the shore",
            "a beautiful beach scene",
            "waves crashing on the beach",
            "a coastal landscape"
        ]
    }
    
    # Copy sample images if they exist
    sample_dir = Path("../data/sample_images")
    if sample_dir.exists():
        import shutil
        for i, img_name in enumerate(["sample_1.jpg", "sample_2.jpg", "sample_3.jpg"], 1):
            src = sample_dir / f"sample_{i}.jpg"
            if src.exists():
                shutil.copy(src, images_dir / img_name)
    
    # Create train/val/test splits
    all_images = list(synthetic_data.keys())
    random.seed(42)
    
    # For real dataset: 6000 train, 1000 val, 1000 test
    # For demo: use all 3 images for each split
    splits = {
        "train": all_images,
        "val": all_images,
        "test": all_images
    }
    
    # Save annotations
    annotations = {}
    for split_name, image_list in splits.items():
        annotations[split_name] = []
        for img_name in image_list:
            for caption in synthetic_data[img_name]:
                annotations[split_name].append({
                    "image": img_name,
                    "caption": caption
                })
    
    # Save to JSON files
    for split_name, data in annotations.items():
        output_file = data_dir / f"{split_name}_captions.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Created {split_name} split: {len(data)} caption-image pairs")
    
    # Create README with instructions for real dataset
    readme = """# Flickr8k Dataset Setup

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
"""
    
    with open(data_dir / "README.md", 'w') as f:
        f.write(readme)
    
    print(f"\n✓ Synthetic dataset created in {data_dir}")
    print(f"✓ {len(all_images)} images with {len(synthetic_data[all_images[0]])} captions each")
    
    return data_dir

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    create_synthetic_dataset()

