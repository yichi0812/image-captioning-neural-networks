"""
RNN-based Image Captioning Model with Attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, encoder_out, decoder_hidden):
        """
        Args:
            encoder_out: (batch_size, num_pixels, encoder_dim)
            decoder_hidden: (batch_size, decoder_dim)
        Returns:
            attention_weights: (batch_size, num_pixels)
            context: (batch_size, encoder_dim)
        """
        # Transform encoder output
        att1 = self.encoder_att(encoder_out)  # (batch, num_pixels, attention_dim)
        
        # Transform decoder hidden state
        att2 = self.decoder_att(decoder_hidden)  # (batch, attention_dim)
        att2 = att2.unsqueeze(1)  # (batch, 1, attention_dim)
        
        # Calculate attention scores
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)  # (batch, num_pixels)
        
        # Apply softmax to get attention weights
        alpha = self.softmax(att)  # (batch, num_pixels)
        
        # Calculate context vector
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch, encoder_dim)
        
        return alpha, context


class LSTMDecoder(nn.Module):
    """
    LSTM Decoder with Attention for Image Captioning
    """
    def __init__(self, embed_dim, decoder_dim, vocab_size, encoder_dim, 
                 attention_dim, dropout=0.5):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        
        # Attention mechanism
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # LSTM cell
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        
        # Linear layers to initialize LSTM state
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        
        # Output layer
        self.fc = nn.Linear(decoder_dim, vocab_size)
        
        # Gate for context
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        
    def init_hidden_state(self, encoder_out):
        """
        Initialize LSTM hidden state
        Args:
            encoder_out: (batch_size, num_pixels, encoder_dim)
        Returns:
            h, c: Initial hidden and cell states
        """
        mean_encoder_out = encoder_out.mean(dim=1)  # (batch, encoder_dim)
        h = self.init_h(mean_encoder_out)  # (batch, decoder_dim)
        c = self.init_c(mean_encoder_out)  # (batch, decoder_dim)
        return h, c
    
    def forward(self, encoder_out, captions, caption_lengths):
        """
        Forward pass
        Args:
            encoder_out: (batch_size, num_pixels, encoder_dim)
            captions: (batch_size, max_length)
            caption_lengths: (batch_size,)
        Returns:
            predictions: (batch_size, max_length, vocab_size)
            alphas: (batch_size, max_length, num_pixels)
        """
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)
        max_length = captions.size(1)
        
        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)
        
        # Embedding
        embeddings = self.embedding(captions)  # (batch, max_length, embed_dim)
        
        # Storage for predictions and attention weights
        predictions = torch.zeros(batch_size, max_length, self.vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max_length, num_pixels).to(encoder_out.device)
        
        # Decode step by step
        for t in range(max_length):
            # Get attention-weighted context
            alpha, context = self.attention(encoder_out, h)
            
            # Gating mechanism
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context
            
            # LSTM input: concatenate embedding and context
            lstm_input = torch.cat([embeddings[:, t, :], gated_context], dim=1)
            
            # LSTM step
            h, c = self.lstm_cell(lstm_input, (h, c))
            
            # Predict next word
            preds = self.fc(self.dropout(h))
            
            predictions[:, t, :] = preds
            alphas[:, t, :] = alpha
        
        return predictions, alphas
    
    def generate_caption(self, encoder_out, vocab, max_length=50, start_token_idx=1, end_token_idx=2):
        """
        Generate caption using greedy decoding
        Args:
            encoder_out: (1, num_pixels, encoder_dim)
            vocab: Vocabulary object
            max_length: Maximum caption length
        Returns:
            caption: Generated caption string
            alphas: Attention weights
        """
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)
        
        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)
        
        # Start with START token
        word = torch.tensor([start_token_idx]).to(encoder_out.device)
        
        # Storage
        caption_indices = []
        alphas_list = []
        
        for t in range(max_length):
            # Get embedding
            embedding = self.embedding(word)  # (1, embed_dim)
            
            # Get attention-weighted context
            alpha, context = self.attention(encoder_out, h)
            alphas_list.append(alpha.cpu().detach())
            
            # Gating mechanism
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context
            
            # LSTM input
            lstm_input = torch.cat([embedding, gated_context], dim=1)
            
            # LSTM step
            h, c = self.lstm_cell(lstm_input, (h, c))
            
            # Predict next word
            preds = self.fc(h)  # (1, vocab_size)
            word = preds.argmax(dim=1)  # (1,)
            
            word_idx = word.item()
            caption_indices.append(word_idx)
            
            # Stop if END token is generated
            if word_idx == end_token_idx:
                break
        
        # Convert to caption
        caption = vocab.decode(caption_indices)
        alphas = torch.stack(alphas_list, dim=1).squeeze(0)  # (length, num_pixels)
        
        return caption, alphas


class RNNCaptioningModel(nn.Module):
    """
    Complete RNN-based Image Captioning Model
    """
    def __init__(self, feature_extractor, vocab_size, embed_dim=512, 
                 decoder_dim=512, attention_dim=512, dropout=0.5):
        super().__init__()
        
        self.feature_extractor = feature_extractor
        self.encoder_dim = feature_extractor.feature_dim
        
        self.decoder = LSTMDecoder(
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            vocab_size=vocab_size,
            encoder_dim=self.encoder_dim,
            attention_dim=attention_dim,
            dropout=dropout
        )
    
    def forward(self, images, captions, caption_lengths):
        """
        Forward pass
        Args:
            images: Preprocessed images
            captions: (batch_size, max_length)
            caption_lengths: (batch_size,)
        Returns:
            predictions: (batch_size, max_length, vocab_size)
            alphas: (batch_size, max_length, num_pixels)
        """
        # Extract features
        _, encoder_out = self.feature_extractor(images)  # (batch, num_pixels, encoder_dim)
        
        # Decode
        predictions, alphas = self.decoder(encoder_out, captions, caption_lengths)
        
        return predictions, alphas
    
    def generate_caption(self, image, vocab, max_length=50):
        """
        Generate caption for a single image
        Args:
            image: Preprocessed image tensor
            vocab: Vocabulary object
            max_length: Maximum caption length
        Returns:
            caption: Generated caption string
            alphas: Attention weights
        """
        self.eval()
        with torch.no_grad():
            # Extract features
            _, encoder_out = self.feature_extractor(image)
            
            # Generate caption
            caption, alphas = self.decoder.generate_caption(
                encoder_out, vocab, max_length,
                start_token_idx=vocab.word2idx[vocab.start_token],
                end_token_idx=vocab.word2idx[vocab.end_token]
            )
        
        return caption, alphas


if __name__ == "__main__":
    print("Testing RNN Captioning Model...")
    
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
    
    model = RNNCaptioningModel(
        feature_extractor=feature_extractor,
        vocab_size=vocab_size,
        embed_dim=256,
        decoder_dim=512,
        attention_dim=256
    )
    
    print(f"Model created successfully!")
    print(f"Encoder dim: {model.encoder_dim}")
    print(f"Decoder vocab size: {model.decoder.vocab_size}")
    
    # Test forward pass
    dummy_images = torch.randn(2, 3, 224, 224)
    dummy_captions = torch.randint(0, vocab_size, (2, 20))
    dummy_lengths = torch.tensor([15, 18])
    
    predictions, alphas = model(dummy_images, dummy_captions, dummy_lengths)
    
    print(f"\nForward pass test:")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Alphas shape: {alphas.shape}")
    
    print("\n✓ RNN Model working correctly!")

