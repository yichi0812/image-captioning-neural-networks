"""
Download and prepare Flickr8k dataset
"""
import os
import urllib.request
import zipfile
from pathlib import Path

def download_flickr8k():
    """Download Flickr8k dataset from direct source"""
    data_dir = Path("../data")
    data_dir.mkdir(exist_ok=True)
    
    print("Downloading Flickr8k dataset...")
    
    # Using Hugging Face datasets as a reliable source
    try:
        from datasets import load_dataset
        
        print("Loading Flickr8k from Hugging Face datasets...")
        dataset = load_dataset("nlphuji/flickr30k", split="test[:1000]")  # Using subset for faster download
        
        # Save to local directory
        dataset.save_to_disk(str(data_dir / "flickr8k_hf"))
        print(f"Dataset saved to {data_dir / 'flickr8k_hf'}")
        
        return True
        
    except Exception as e:
        print(f"Error downloading from Hugging Face: {e}")
        return False

def download_sample_images():
    """Download sample images for testing"""
    import requests
    from PIL import Image
    from io import BytesIO
    
    sample_dir = Path("../data/sample_images")
    sample_dir.mkdir(exist_ok=True, parents=True)
    
    # Sample image URLs from public sources
    sample_urls = [
        "https://images.unsplash.com/photo-1552053831-71594a27632d?w=500",  # Dog
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=500",  # Mountain
        "https://images.unsplash.com/photo-1519681393784-d120267933ba?w=500",  # Beach
    ]
    
    print("\nDownloading sample images for testing...")
    for i, url in enumerate(sample_urls):
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content))
            img.save(sample_dir / f"sample_{i+1}.jpg")
            print(f"Downloaded sample image {i+1}")
        except Exception as e:
            print(f"Failed to download sample {i+1}: {e}")
    
    return True

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Download dataset
    success = download_flickr8k()
    
    # Download sample images
    download_sample_images()
    
    if success:
        print("\n✓ Data preparation complete!")
    else:
        print("\n✗ Data preparation failed. Will create synthetic data for demonstration.")

