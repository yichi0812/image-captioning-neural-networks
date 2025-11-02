from pathlib import Path
from src.dataset import get_data_loaders

print("Testing dataset loading...")
data_dir = "/home/ubuntu/image-captioning-academic/data/flickr8k"
print(f"Data directory: {data_dir}")
print(f"Exists: {Path(data_dir).exists()}")

try:
    train_loader, val_loader, test_loader, vocab = get_data_loaders(data_dir, batch_size=2)
    print("Success!")
    print(f"Vocabulary size: {len(vocab)}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

