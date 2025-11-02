"""
Transformer-based Image Captioning Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer with Self-Attention and Cross-Attention
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Self-attention (masked)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Cross-attention (attend to image)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()
    
    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        """
        Args:
            tgt: Target sequence (batch, tgt_len, d_model)
            memory: Encoder output / image features (batch, src_len, d_model)
            tgt_mask: Causal mask for target
            tgt_key_padding_mask: Padding mask for target
        Returns:
            output: (batch, tgt_len, d_model)
            cross_attn_weights: (batch, tgt_len, src_len)
        """
        # Self-attention with residual connection
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, 
                                  key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention with residual connection
        tgt2, cross_attn_weights = self.cross_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed-forward with residual connection
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt, cross_attn_weights


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder for Image Captioning
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, 
                 dim_feedforward=2048, dropout=0.1, max_len=100):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
    
    def generate_square_subsequent_mask(self, sz, device):
        """
        Generate causal mask for decoder
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        """
        Args:
            tgt: Target tokens (batch, tgt_len)
            memory: Image features (batch, num_pixels, d_model)
            tgt_mask: Causal mask
            tgt_key_padding_mask: Padding mask
        Returns:
            output: (batch, tgt_len, vocab_size)
            cross_attn_weights: List of attention weights from each layer
        """
        # Embedding and positional encoding
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        
        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), tgt.device)
        
        # Pass through decoder layers
        cross_attn_weights_list = []
        for layer in self.layers:
            tgt, cross_attn_weights = layer(tgt, memory, tgt_mask, tgt_key_padding_mask)
            cross_attn_weights_list.append(cross_attn_weights)
        
        # Project to vocabulary
        output = self.fc_out(tgt)
        
        return output, cross_attn_weights_list
    
    def generate_caption(self, memory, vocab, max_length=50, start_token_idx=1, 
                        end_token_idx=2, temperature=1.0):
        """
        Generate caption using greedy decoding
        Args:
            memory: Image features (1, num_pixels, d_model)
            vocab: Vocabulary object
            max_length: Maximum caption length
            temperature: Sampling temperature
        Returns:
            caption: Generated caption string
            cross_attn_weights: Attention weights
        """
        device = memory.device
        batch_size = memory.size(0)
        
        # Start with START token
        tgt = torch.tensor([[start_token_idx]], device=device)
        
        caption_indices = []
        all_attn_weights = []
        
        for _ in range(max_length):
            # Forward pass
            output, cross_attn_weights = self.forward(tgt, memory)
            
            # Get last token prediction
            logits = output[:, -1, :] / temperature
            next_token = logits.argmax(dim=-1, keepdim=True)
            
            # Store attention weights from last layer
            all_attn_weights.append(cross_attn_weights[-1][:, -1, :].cpu().detach())
            
            # Add to sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            token_idx = next_token.item()
            caption_indices.append(token_idx)
            
            # Stop if END token
            if token_idx == end_token_idx:
                break
        
        # Convert to caption
        caption = vocab.decode(caption_indices)
        
        # Stack attention weights
        if all_attn_weights:
            attn_weights = torch.stack(all_attn_weights, dim=0).squeeze(1)  # (length, num_pixels)
        else:
            attn_weights = None
        
        return caption, attn_weights


class TransformerCaptioningModel(nn.Module):
    """
    Complete Transformer-based Image Captioning Model
    """
    def __init__(self, feature_extractor, vocab_size, d_model=512, nhead=8, 
                 num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        self.feature_extractor = feature_extractor
        self.encoder_dim = feature_extractor.feature_dim
        
        # Project image features to d_model if needed
        if self.encoder_dim != d_model:
            self.feature_proj = nn.Linear(self.encoder_dim, d_model)
        else:
            self.feature_proj = nn.Identity()
        
        # Transformer decoder
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
    
    def forward(self, images, captions, caption_lengths=None):
        """
        Forward pass
        Args:
            images: Preprocessed images
            captions: (batch_size, max_length)
            caption_lengths: (batch_size,) - optional
        Returns:
            predictions: (batch_size, max_length, vocab_size)
            cross_attn_weights: List of attention weights
        """
        # Extract features
        _, encoder_out = self.feature_extractor(images)  # (batch, num_pixels, encoder_dim)
        
        # Project features
        memory = self.feature_proj(encoder_out)  # (batch, num_pixels, d_model)
        
        # Create padding mask if lengths provided
        tgt_key_padding_mask = None
        if caption_lengths is not None:
            max_len = captions.size(1)
            tgt_key_padding_mask = torch.arange(max_len, device=captions.device)[None, :] >= caption_lengths[:, None]
        
        # Decode
        predictions, cross_attn_weights = self.decoder(captions, memory, 
                                                       tgt_key_padding_mask=tgt_key_padding_mask)
        
        return predictions, cross_attn_weights
    
    def generate_caption(self, image, vocab, max_length=50, temperature=1.0):
        """
        Generate caption for a single image
        Args:
            image: Preprocessed image tensor
            vocab: Vocabulary object
            max_length: Maximum caption length
            temperature: Sampling temperature
        Returns:
            caption: Generated caption string
            attn_weights: Attention weights
        """
        self.eval()
        with torch.no_grad():
            # Extract features
            _, encoder_out = self.feature_extractor(image)
            memory = self.feature_proj(encoder_out)
            
            # Generate caption
            caption, attn_weights = self.decoder.generate_caption(
                memory, vocab, max_length,
                start_token_idx=vocab.word2idx[vocab.start_token],
                end_token_idx=vocab.word2idx[vocab.end_token],
                temperature=temperature
            )
        
        return caption, attn_weights


if __name__ == "__main__":
    print("Testing Transformer Captioning Model...")
    
    # Create dummy feature extractor
    class DummyFeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_dim = 768
        
        def forward(self, x):
            batch_size = 2
            num_pixels = 50
            pooled = torch.randn(batch_size, self.feature_dim)
            spatial = torch.randn(batch_size, num_pixels, self.feature_dim)
            return pooled, spatial
    
    feature_extractor = DummyFeatureExtractor()
    vocab_size = 100
    
    model = TransformerCaptioningModel(
        feature_extractor=feature_extractor,
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_layers=3,
        dim_feedforward=2048
    )
    
    print(f"Model created successfully!")
    print(f"Encoder dim: {model.encoder_dim}")
    print(f"Decoder vocab size: {model.decoder.vocab_size}")
    
    # Test forward pass
    dummy_images = torch.randn(2, 3, 224, 224)
    dummy_captions = torch.randint(0, vocab_size, (2, 20))
    dummy_lengths = torch.tensor([15, 18])
    
    predictions, cross_attn_weights = model(dummy_images, dummy_captions, dummy_lengths)
    
    print(f"\nForward pass test:")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Number of attention layers: {len(cross_attn_weights)}")
    print(f"Attention shape (last layer): {cross_attn_weights[-1].shape}")
    
    print("\n✓ Transformer Model working correctly!")

