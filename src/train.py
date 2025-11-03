"""
Training pipeline for image captioning models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import time

from feature_extractor import get_feature_extractor
from dataset import get_data_loaders, Vocabulary
from rnn_model import RNNCaptioningModel
from transformer_model import TransformerCaptioningModel


class Trainer:
    """
    Trainer for image captioning models
    """
    def __init__(self, model, train_loader, val_loader, vocab, config, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.config = config
        self.device = device
        
        # Loss function (ignore padding)
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx[vocab.pad_token])
        
        # Optimizer
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['learning_rate']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for batch in progress_bar:
            # Get data
            images = batch['images']
            captions = batch['captions_encoded'].to(self.device)
            caption_lengths = batch['caption_lengths'].to(self.device)
            
            # Preprocess images
            image_tensors = []
            for img in images:
                img_tensor = self.model.feature_extractor.preprocess(img)
                image_tensors.append(img_tensor)
            image_tensors = torch.cat(image_tensors, dim=0).to(self.device)
            
            # Forward pass
            # For training, we predict next token, so input is captions[:-1], target is captions[1:]
            input_captions = captions[:, :-1]
            target_captions = captions[:, 1:]
            
            predictions, _ = self.model(image_tensors, input_captions, caption_lengths - 1)
            
            # Calculate loss
            # Reshape for cross entropy: (batch * seq_len, vocab_size)
            predictions = predictions.reshape(-1, predictions.size(-1))
            targets = target_captions.reshape(-1)
            
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Get data
                images = batch['images']
                captions = batch['captions_encoded'].to(self.device)
                caption_lengths = batch['caption_lengths'].to(self.device)
                
                # Preprocess images
                image_tensors = []
                for img in images:
                    img_tensor = self.model.feature_extractor.preprocess(img)
                    image_tensors.append(img_tensor)
                image_tensors = torch.cat(image_tensors, dim=0).to(self.device)
                
                # Forward pass
                input_captions = captions[:, :-1]
                target_captions = captions[:, 1:]
                
                predictions, _ = self.model(image_tensors, input_captions, caption_lengths - 1)
                
                # Calculate loss
                predictions = predictions.reshape(-1, predictions.size(-1))
                targets = target_captions.reshape(-1)
                
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, num_epochs, save_dir):
        """
        Train the model
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            epoch_time = time.time() - start_time
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"Time: {epoch_time:.2f}s")
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': self.history,
                'config': self.config
            }
            
            # Save latest checkpoint
            torch.save(checkpoint, save_dir / 'checkpoint_latest.pth')
            
            # Save best model
            if val_loss < self.best_val_loss:
                print(f"Validation loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}")
                self.best_val_loss = val_loss
                torch.save(checkpoint, save_dir / 'checkpoint_best.pth')
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= self.config['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model
        torch.save(checkpoint, save_dir / 'checkpoint_final.pth')
        
        # Save training history
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def train_rnn_model(data_dir, save_dir, config):
    """Train RNN-based model"""
    print("=" * 60)
    print("Training RNN-based Image Captioning Model")
    print("=" * 60)
    
    # Get data loaders
    train_loader, val_loader, test_loader, vocab = get_data_loaders(
        data_dir=data_dir,
        batch_size=config['batch_size']
    )
    
    # Create feature extractor
    feature_extractor = get_feature_extractor(
        model_type=config['feature_extractor'],
        freeze=True
    )
    
    # Create model
    model = RNNCaptioningModel(
        feature_extractor=feature_extractor,
        vocab_size=len(vocab),
        embed_dim=config['embed_dim'],
        decoder_dim=config['decoder_dim'],
        attention_dim=config['attention_dim'],
        dropout=config['dropout']
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        config=config,
        device=config['device']
    )
    
    # Train
    trainer.train(num_epochs=config['num_epochs'], save_dir=save_dir)
    
    return model, vocab


def train_transformer_model(data_dir, save_dir, config):
    """Train Transformer-based model"""
    print("=" * 60)
    print("Training Transformer-based Image Captioning Model")
    print("=" * 60)
    
    # Get data loaders
    train_loader, val_loader, test_loader, vocab = get_data_loaders(
        data_dir=data_dir,
        batch_size=config['batch_size']
    )
    
    # Create feature extractor
    feature_extractor = get_feature_extractor(
        model_type=config['feature_extractor'],
        freeze=True
    )
    
    # Create model
    model = TransformerCaptioningModel(
        feature_extractor=feature_extractor,
        vocab_size=len(vocab),
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        config=config,
        device=config['device']
    )
    
    # Train
    trainer.train(num_epochs=config['num_epochs'], save_dir=save_dir)
    
    return model, vocab


if __name__ == "__main__":
    # Configuration
    rnn_config = {
        'feature_extractor': 'clip',
        'batch_size': 4,
        'embed_dim': 256,
        'decoder_dim': 512,
        'attention_dim': 256,
        'dropout': 0.3,
        'learning_rate': 3e-4,
        'num_epochs': 20,
        'grad_clip': 5.0,
        'early_stopping_patience': 5,
        'device': 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    transformer_config = {
        'feature_extractor': 'clip',
        'batch_size': 4,
        'd_model': 512,
        'nhead': 8,
        'num_layers': 3,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'learning_rate': 1e-4,
        'num_epochs': 20,
        'grad_clip': 5.0,
        'early_stopping_patience': 5,
        'device': 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    data_dir = "/home/ubuntu/image-captioning-academic/data"
    
    # Train RNN model
    print("\n" + "=" * 60)
    print("TRAINING RNN MODEL")
    print("=" * 60 + "\n")
    rnn_model, vocab = train_rnn_model(
        data_dir=data_dir,
        save_dir="../models/rnn",
        config=rnn_config
    )
    
    # Train Transformer model
    print("\n" + "=" * 60)
    print("TRAINING TRANSFORMER MODEL")
    print("=" * 60 + "\n")
    transformer_model, vocab = train_transformer_model(
        data_dir=data_dir,
        save_dir="../models/transformer",
        config=transformer_config
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

