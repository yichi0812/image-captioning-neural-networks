"""
Dataset classes for image captioning
"""
import json
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pickle

class Vocabulary:
    """Vocabulary for caption tokens"""
    
    def __init__(self, freq_threshold=2):
        self.freq_threshold = freq_threshold
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.start_token = "<START>"
        self.end_token = "<END>"
        self.unk_token = "<UNK>"
        
        # Initialize with special tokens
        self.word2idx = {
            self.pad_token: 0,
            self.start_token: 1,
            self.end_token: 2,
            self.unk_token: 3
        }
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
    def build_vocabulary(self, captions):
        """Build vocabulary from list of captions"""
        # Count word frequencies
        for caption in captions:
            tokens = self.tokenize(caption)
            self.word_freq.update(tokens)
        
        # Add words that meet frequency threshold
        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.freq_threshold:
                if word not in self.word2idx:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1
        
        print(f"Vocabulary size: {len(self.word2idx)}")
        print(f"Most common words: {self.word_freq.most_common(10)}")
    
    def tokenize(self, text):
        """Simple tokenization"""
        return text.lower().strip().split()
    
    def encode(self, text, add_special_tokens=True):
        """Convert text to token indices"""
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = [self.start_token] + tokens + [self.end_token]
        
        indices = []
        for token in tokens:
            if token in self.word2idx:
                indices.append(self.word2idx[token])
            else:
                indices.append(self.word2idx[self.unk_token])
        
        return indices
    
    def decode(self, indices, skip_special_tokens=True):
        """Convert token indices to text"""
        tokens = []
        for idx in indices:
            if idx in self.idx2word:
                token = self.idx2word[idx]
                if skip_special_tokens and token in [self.pad_token, self.start_token, self.end_token]:
                    continue
                tokens.append(token)
        
        return ' '.join(tokens)
    
    def __len__(self):
        return len(self.word2idx)
    
    def save(self, path):
        """Save vocabulary to file"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_freq': self.word_freq,
                'freq_threshold': self.freq_threshold
            }, f)
    
    def load(self, path):
        """Load vocabulary from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.word_freq = data['word_freq']
            self.freq_threshold = data['freq_threshold']


class CaptionDataset(Dataset):
    """Dataset for image captioning"""
    
    def __init__(self, data_file, images_dir, vocabulary=None, transform=None, max_length=50):
        """
        Args:
            data_file: JSON file with captions
            images_dir: Directory containing images
            vocabulary: Vocabulary object (if None, will be built)
            transform: Image transformation function
            max_length: Maximum caption length
        """
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.max_length = max_length
        
        # Load captions
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} caption-image pairs from {data_file}")
        
        # Build or use existing vocabulary
        if vocabulary is None:
            self.vocab = Vocabulary(freq_threshold=1)  # Lower threshold for small dataset
            all_captions = [item['caption'] for item in self.data]
            self.vocab.build_vocabulary(all_captions)
        else:
            self.vocab = vocabulary
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get image and caption pair"""
        item = self.data[idx]
        
        # Load image
        img_path = self.images_dir / item['image']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='white')
        
        # Encode caption
        caption = item['caption']
        caption_encoded = self.vocab.encode(caption)
        
        # Pad or truncate caption
        if len(caption_encoded) > self.max_length:
            caption_encoded = caption_encoded[:self.max_length]
        
        # Convert to tensor
        caption_tensor = torch.zeros(self.max_length, dtype=torch.long)
        caption_tensor[:len(caption_encoded)] = torch.tensor(caption_encoded)
        
        # Get actual length for masking
        caption_length = len(caption_encoded)
        
        return {
            'image': image,
            'image_path': str(img_path),
            'caption': caption,
            'caption_encoded': caption_tensor,
            'caption_length': caption_length
        }


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    images = [item['image'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    captions = [item['caption'] for item in batch]
    captions_encoded = torch.stack([item['caption_encoded'] for item in batch])
    caption_lengths = torch.tensor([item['caption_length'] for item in batch])
    
    return {
        'images': images,
        'image_paths': image_paths,
        'captions': captions,
        'captions_encoded': captions_encoded,
        'caption_lengths': caption_lengths
    }


def get_data_loaders(data_dir, batch_size=32, num_workers=0):
    """
    Create data loaders for train, val, and test sets
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size
        num_workers: Number of worker processes
    
    Returns:
        train_loader, val_loader, test_loader, vocabulary
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    
    # Create train dataset and build vocabulary
    train_dataset = CaptionDataset(
        data_file=data_dir / "train_captions.json",
        images_dir=images_dir,
        vocabulary=None  # Will build vocabulary
    )
    
    vocab = train_dataset.vocab
    
    # Save vocabulary
    vocab_path = data_dir / "vocabulary.pkl"
    vocab.save(vocab_path)
    print(f"Vocabulary saved to {vocab_path}")
    
    # Create val and test datasets with same vocabulary
    val_dataset = CaptionDataset(
        data_file=data_dir / "val_captions.json",
        images_dir=images_dir,
        vocabulary=vocab
    )
    
    test_dataset = CaptionDataset(
        data_file=data_dir / "test_captions.json",
        images_dir=images_dir,
        vocabulary=vocab
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader, vocab


if __name__ == "__main__":
    # Test dataset
    print("Testing dataset...")
    
    data_dir = Path("../data/flickr8k")
    
    train_loader, val_loader, test_loader, vocab = get_data_loaders(
        data_dir=data_dir,
        batch_size=2
    )
    
    print(f"\nDataset statistics:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Vocabulary size: {len(vocab)}")
    
    # Test one batch
    print("\nTesting one batch...")
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Number of images: {len(batch['images'])}")
    print(f"Captions encoded shape: {batch['captions_encoded'].shape}")
    print(f"Caption lengths: {batch['caption_lengths']}")
    print(f"\nSample caption: {batch['captions'][0]}")
    print(f"Encoded: {batch['captions_encoded'][0][:10]}")
    print(f"Decoded: {vocab.decode(batch['captions_encoded'][0].tolist())}")
    
    print("\n✓ Dataset working correctly!")

