"""
Data Preparation Script for Image Captioning
Downloads and prepares Flickr8k or MS COCO dataset
"""

import argparse
import os
import urllib.request
import zipfile
from pathlib import Path
import json


def download_flickr8k(data_dir):
    """Download Flickr8k dataset"""
    print("=" * 60)
    print("Flickr8k Dataset Download")
    print("=" * 60)
    
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    print("\nNote: Flickr8k dataset requires manual download from Kaggle")
    print("Please follow these steps:")
    print("\n1. Visit: https://www.kaggle.com/datasets/adityajn105/flickr8k")
    print("2. Download the dataset")
    print(f"3. Extract to: {data_path.absolute()}")
    print("\nExpected structure:")
    print(f"  {data_path}/Images/")
    print(f"  {data_path}/captions.txt")
    
    # Check if data already exists
    if (data_path / "Images").exists() and (data_path / "captions.txt").exists():
        print("\n✓ Dataset found!")
        return True
    else:
        print("\n✗ Dataset not found. Please download manually.")
        return False


def prepare_flickr8k(data_dir):
    """Prepare Flickr8k dataset for training"""
    print("\n" + "=" * 60)
    print("Preparing Flickr8k Dataset")
    print("=" * 60)
    
    data_path = Path(data_dir)
    images_dir = data_path / "Images"
    captions_file = data_path / "captions.txt"
    
    if not images_dir.exists() or not captions_file.exists():
        print("Error: Dataset not found. Please download first.")
        return False
    
    # Read captions
    print("\nReading captions...")
    captions_dict = {}
    
    with open(captions_file, 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                image_name, caption = parts
                if image_name not in captions_dict:
                    captions_dict[image_name] = []
                captions_dict[image_name].append(caption.strip())
    
    print(f"Found {len(captions_dict)} images with captions")
    
    # Split into train/val/test (80/10/10)
    image_names = list(captions_dict.keys())
    total = len(image_names)
    
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    
    train_images = image_names[:train_size]
    val_images = image_names[train_size:train_size + val_size]
    test_images = image_names[train_size + val_size:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_images)} images")
    print(f"  Val:   {len(val_images)} images")
    print(f"  Test:  {len(test_images)} images")
    
    # Save splits in the format expected by dataset.py: list of {image, caption} dicts
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    for split_name, image_list in splits.items():
        # Convert to list format: [{"image": "img.jpg", "caption": "caption text"}, ...]
        annotations = []
        for img_name in image_list:
            for caption in captions_dict[img_name]:
                annotations.append({
                    "image": img_name,
                    "caption": caption
                })
        
        output_file = data_path / f"{split_name}_captions.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2)
        print(f"Saved {split_name} split to {output_file} ({len(annotations)} caption-image pairs)")
    
    print("\n✓ Dataset preparation complete!")
    return True


def download_coco(data_dir):
    """Instructions for downloading MS COCO dataset"""
    print("=" * 60)
    print("MS COCO Dataset Download")
    print("=" * 60)
    
    print("\nMS COCO dataset is large (~13GB for train, ~6GB for val)")
    print("Please download manually from: http://cocodataset.org/#download")
    print("\nRequired files:")
    print("  - 2017 Train images")
    print("  - 2017 Val images")
    print("  - 2017 Train/Val annotations")
    
    data_path = Path(data_dir)
    print(f"\nExtract to: {data_path.absolute()}")
    print("\nExpected structure:")
    print(f"  {data_path}/train2017/")
    print(f"  {data_path}/val2017/")
    print(f"  {data_path}/annotations/")


def main():
    parser = argparse.ArgumentParser(description='Prepare image captioning dataset')
    parser.add_argument('--dataset', type=str, default='flickr8k',
                       choices=['flickr8k', 'coco'],
                       help='Dataset to prepare')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory to store dataset')
    parser.add_argument('--download', action='store_true',
                       help='Download dataset (shows instructions)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Image Captioning Dataset Preparation")
    print("=" * 60)
    
    if args.dataset == 'flickr8k':
        if args.download:
            download_flickr8k(args.data_dir)
        prepare_flickr8k(args.data_dir)
    elif args.dataset == 'coco':
        download_coco(args.data_dir)
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Ensure dataset is properly downloaded")
    print("  2. Run: python src/train.py --model rnn")
    print("  3. Run: python src/train.py --model transformer")
    print("=" * 60)


if __name__ == '__main__':
    main()

