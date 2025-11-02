"""
Run evaluation on trained models
"""
import torch
from pathlib import Path
import pickle

from feature_extractor import get_feature_extractor
from dataset import get_data_loaders, Vocabulary
from rnn_model import RNNCaptioningModel
from transformer_model import TransformerCaptioningModel
from evaluate import evaluate_model, compare_models


def load_model(model_type, checkpoint_path, vocab, device='cpu'):
    """
    Load a trained model from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create feature extractor
    feature_extractor = get_feature_extractor(
        model_type=config['feature_extractor'],
        freeze=True
    )
    
    # Create model
    if model_type == 'rnn':
        model = RNNCaptioningModel(
            feature_extractor=feature_extractor,
            vocab_size=len(vocab),
            embed_dim=config['embed_dim'],
            decoder_dim=config['decoder_dim'],
            attention_dim=config['attention_dim'],
            dropout=config['dropout']
        )
    elif model_type == 'transformer':
        model = TransformerCaptioningModel(
            feature_extractor=feature_extractor,
            vocab_size=len(vocab),
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


if __name__ == "__main__":
    # Configuration
    data_dir = Path("../data/flickr8k")
    rnn_checkpoint = Path("../models/rnn/checkpoint_best.pth")
    transformer_checkpoint = Path("../models/transformer/checkpoint_best.pth")
    output_dir = Path("../outputs")
    output_dir.mkdir(exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load vocabulary
    vocab_path = data_dir / "vocabulary.pkl"
    vocab = Vocabulary()
    vocab.load(vocab_path)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Get test data loader
    _, _, test_loader, _ = get_data_loaders(
        data_dir=data_dir,
        batch_size=4
    )
    
    # Load models
    print("\nLoading RNN model...")
    rnn_model = load_model('rnn', rnn_checkpoint, vocab, device)
    
    print("Loading Transformer model...")
    transformer_model = load_model('transformer', transformer_checkpoint, vocab, device)
    
    # Evaluate models
    print("\n" + "=" * 60)
    print("EVALUATING MODELS")
    print("=" * 60)
    
    print("\nEvaluating RNN model...")
    rnn_results = evaluate_model(rnn_model, test_loader, vocab, device)
    
    print("\nEvaluating Transformer model...")
    transformer_results = evaluate_model(transformer_model, test_loader, vocab, device)
    
    # Compare models
    rnn_metrics, transformer_metrics = compare_models(
        rnn_results,
        transformer_results,
        output_file=output_dir / "evaluation_results.json"
    )
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)

