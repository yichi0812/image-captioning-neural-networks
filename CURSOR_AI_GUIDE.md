# Cursor AI Guide - Image Captioning Project

Complete reference document for understanding and working with this image captioning project using Cursor AI.

---

## 📋 Project Overview

**Goal:** Build an automated image captioning system that generates natural language descriptions for images using deep learning.

**Approach:** 
- Use pre-trained CLIP for image feature extraction
- Implement two decoder architectures: RNN with attention and Transformer
- Train on Flickr8k dataset (8,091 images with 5 captions each)
- Evaluate using BLEU, METEOR, CIDEr, and ROUGE-L metrics
- Compare RNN vs Transformer performance

---

## 🏗️ Architecture Details

### Overall Pipeline

```
Input Image 
    ↓
CLIP Feature Extractor (ViT-B/32)
    ↓
Visual Features (512-dim embeddings)
    ↓
Decoder (RNN or Transformer)
    ↓
Generated Caption (word by word)
```

### 1. Feature Extractor (CLIP)

**File:** `src/feature_extractor.py`

**What it does:**
- Takes input image (any size)
- Resizes to 224x224
- Extracts visual features using pre-trained CLIP ViT-B/32
- Outputs 512-dimensional feature vector

**Why CLIP instead of training CNN from scratch:**
- Pre-trained on 400M image-text pairs
- Better semantic understanding
- No need for expensive CNN training
- Faster convergence

**Key code:**
```python
class CLIPFeatureExtractor:
    def __init__(self):
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def forward(self, images):
        # Returns [batch_size, 512] feature vectors
        return self.model(pixel_values=inputs.pixel_values).pooler_output
```

---

### 2. RNN Decoder with Attention

**File:** `src/rnn_model.py`

**Architecture:**
```
Visual Features (512-dim)
    ↓
Embedding Layer (vocab_size → 256-dim)
    ↓
LSTM (2 layers, 512 hidden units)
    ↓
Bahdanau Attention Mechanism
    ↓
Output Layer (512 → vocab_size)
    ↓
Softmax → Word Prediction
```

**Key Components:**

1. **Embedding Layer**
   - Converts word indices to dense vectors
   - Dimension: vocab_size → 256

2. **LSTM (Long Short-Term Memory)**
   - 2 layers
   - 512 hidden units per layer
   - Dropout: 0.3
   - Processes sequence one word at a time

3. **Bahdanau Attention**
   - Learns where to "look" in the image
   - Attention dimension: 256
   - Computes attention weights over image features
   - Produces context vector for each time step

4. **Output Layer**
   - Linear layer: 512 → vocab_size
   - Softmax to get word probabilities

**Training Config:**
```python
{
    'embed_dim': 256,
    'hidden_dim': 512,
    'attention_dim': 256,
    'num_layers': 2,
    'dropout': 0.3,
    'learning_rate': 3e-4,
    'batch_size': 32,
    'epochs': 20
}
```

---

### 3. Transformer Decoder

**File:** `src/transformer_model.py`

**Architecture:**
```
Visual Features (512-dim)
    ↓
Positional Encoding
    ↓
Transformer Decoder (6 layers)
    ├── Multi-Head Self-Attention (8 heads)
    ├── Multi-Head Cross-Attention (8 heads)
    └── Feed-Forward Network (2048-dim)
    ↓
Output Layer (512 → vocab_size)
    ↓
Softmax → Word Prediction
```

**Key Components:**

1. **Positional Encoding**
   - Adds position information to embeddings
   - Sine/cosine functions
   - Allows model to understand word order

2. **Multi-Head Attention**
   - 8 attention heads
   - Each head learns different aspects
   - Self-attention: word-to-word relationships
   - Cross-attention: word-to-image relationships

3. **Feed-Forward Network**
   - 2 linear layers
   - Hidden dimension: 2048
   - GELU activation
   - Dropout: 0.1

4. **Layer Normalization**
   - Stabilizes training
   - Applied before each sub-layer

**Training Config:**
```python
{
    'model_dim': 512,
    'num_heads': 8,
    'num_layers': 3,
    'ff_dim': 2048,
    'dropout': 0.1,
    'learning_rate': 1e-4,
    'batch_size': 32,
    'epochs': 20
}
```

**Why Transformer is better:**
- Parallel processing (faster training)
- Better long-range dependencies
- More effective attention mechanism
- State-of-the-art performance

---

## 📊 Dataset Details

**File:** `src/dataset.py`

### Flickr8k Dataset

- **Total images:** 8,091
- **Captions per image:** 5
- **Total captions:** 40,455
- **Vocabulary size:** ~8,500 words (with freq threshold = 5)

### Data Split

```
Training:   6,472 images (32,360 captions) - 80%
Validation:   809 images  (4,045 captions) - 10%
Test:         810 images  (4,050 captions) - 10%
```

### Data Processing

1. **Caption Preprocessing:**
   ```python
   - Convert to lowercase
   - Remove punctuation
   - Add <start> and <end> tokens
   - Tokenize into words
   - Convert to indices using vocabulary
   ```

2. **Image Preprocessing:**
   ```python
   - Resize to 224x224
   - Normalize: mean=[0.48145466, 0.4578275, 0.40821073]
                std=[0.26862954, 0.26130258, 0.27577711]
   - Convert to tensor
   ```

3. **Vocabulary Building:**
   ```python
   - Special tokens: <pad>=0, <start>=1, <end>=2, <unk>=3
   - Frequency threshold: 5 (words appearing < 5 times → <unk>)
   - Final vocab size: ~8,500 words
   ```

---

## 🎯 Training Process

**File:** `src/train.py`

### Training Loop

```python
for epoch in range(num_epochs):
    # Training phase
    for batch in train_loader:
        images, captions, lengths = batch
        
        # Forward pass
        features = feature_extractor(images)
        outputs = model(features, captions, lengths)
        
        # Compute loss (cross-entropy)
        loss = criterion(outputs, captions)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        
        # Update weights
        optimizer.step()
    
    # Validation phase
    val_loss = validate(model, val_loader)
    
    # Save checkpoint if improved
    if val_loss < best_val_loss:
        save_checkpoint(model, optimizer, epoch, val_loss)
    
    # Early stopping
    if no_improvement_for_5_epochs:
        break
```

### Loss Function

**Cross-Entropy Loss:**
```python
loss = -Σ log(P(word_i | word_1, ..., word_{i-1}, image))
```

- Measures how well predicted word distribution matches true word
- Lower is better
- Typical range: 3.5 (epoch 1) → 1.5-2.0 (epoch 20)

### Optimizer

**Adam Optimizer:**
- RNN: learning rate = 3e-4
- Transformer: learning rate = 1e-4
- Beta1 = 0.9, Beta2 = 0.999
- Weight decay = 1e-5

### Learning Rate Schedule

**ReduceLROnPlateau:**
- Reduces LR when validation loss plateaus
- Factor: 0.5 (halves the learning rate)
- Patience: 3 epochs
- Minimum LR: 1e-6

### Early Stopping

- Patience: 5 epochs
- Stops training if no improvement in validation loss
- Prevents overfitting

---

## 📈 Evaluation Metrics

**File:** `src/evaluate.py`

### 1. BLEU (Bilingual Evaluation Understudy)

**What it measures:** N-gram overlap between generated and reference captions

**Variants:**
- BLEU-1: Unigram precision
- BLEU-2: Bigram precision
- BLEU-3: Trigram precision
- BLEU-4: 4-gram precision (most important)

**Expected scores:**
- BLEU-1: 0.55-0.65
- BLEU-4: 0.20-0.30

**Interpretation:**
- Higher is better
- 0.30+ is good for image captioning

### 2. METEOR (Metric for Evaluation of Translation with Explicit ORdering)

**What it measures:** 
- Unigram matching
- Considers synonyms
- Accounts for word order

**Expected score:** 0.22-0.28

**Interpretation:**
- Higher is better
- Better than BLEU for semantic similarity

### 3. CIDEr (Consensus-based Image Description Evaluation)

**What it measures:**
- TF-IDF weighted n-gram matching
- Designed specifically for image captioning
- Considers consensus across multiple references

**Expected score:** 0.70-0.90

**Interpretation:**
- Higher is better
- Most reliable metric for image captioning

### 4. ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation - Longest Common Subsequence)

**What it measures:**
- Longest common subsequence
- Sentence-level structure

**Expected score:** 0.45-0.55

**Interpretation:**
- Higher is better
- Good for fluency evaluation

---

## 🎨 Attention Visualization

**File:** `src/visualize_attention.py`

### How Attention Works

1. **Compute attention weights** for each word
2. **Overlay weights** on image regions
3. **Generate heatmap** showing where model "looks"

### Visualization Process

```python
1. Load image
2. Generate caption word by word
3. For each word:
   - Get attention weights (49 values for 7x7 grid)
   - Reshape to 7x7 spatial map
   - Upsample to original image size
   - Create heatmap overlay
4. Save visualization
```

### Interpretation

- **Red/Yellow regions:** High attention (model is looking here)
- **Blue regions:** Low attention (model ignores this)
- **Example:** When generating "dog", attention focuses on dog region

---

## 🚀 Training Timeline

### Expected Progress (M3 Pro, batch size 128)

| Epoch | RNN Val Loss | Transformer Val Loss | Time |
|-------|--------------|---------------------|------|
| 1 | 2.75 | 2.85 | 10 min |
| 5 | 2.20 | 2.10 | 50 min |
| 10 | 1.90 | 1.75 | 100 min |
| 15 | 1.75 | 1.55 | 150 min |
| 20 | 1.65 | 1.45 | 200 min |

### Total Training Time

- **RNN:** ~2-2.5 hours
- **Transformer:** ~2.5-3 hours
- **Total:** ~5 hours

---

## 💻 Cursor AI Prompts

### Understanding the Code

```
@workspace Explain how the attention mechanism works in src/rnn_model.py
```

```
@workspace What is the difference between the RNN and Transformer decoders?
```

```
@workspace How does the training loop in src/train.py handle gradient clipping?
```

### Debugging

```
@workspace Why might I get "CUDA out of memory" error during training?
```

```
@workspace The validation loss is not decreasing. What could be wrong?
```

```
@workspace How can I reduce the model size to fit in my GPU memory?
```

### Modifications

```
@workspace How can I add beam search for better caption generation?
```

```
@workspace Can you help me implement a larger Transformer with 6 layers?
```

```
@workspace How do I modify the code to use a different dataset?
```

### Optimization

```
@workspace How can I speed up training on my M3 Pro?
```

```
@workspace What batch size should I use for 16GB GPU memory?
```

```
@workspace Can you add mixed precision training to src/train.py?
```

---

## 🔍 Key Files Explained

### 1. `src/feature_extractor.py`
**Purpose:** Extract visual features from images using CLIP  
**Key class:** `CLIPFeatureExtractor`  
**Input:** PIL Image or batch of images  
**Output:** 512-dim feature vectors  

### 2. `src/dataset.py`
**Purpose:** Load and preprocess Flickr8k dataset  
**Key classes:** `Vocabulary`, `CaptionDataset`  
**Handles:** Tokenization, vocabulary building, data loading  

### 3. `src/rnn_model.py`
**Purpose:** RNN decoder with Bahdanau attention  
**Key classes:** `Attention`, `RNNCaptioningModel`  
**Architecture:** LSTM + Attention  

### 4. `src/transformer_model.py`
**Purpose:** Transformer decoder  
**Key class:** `TransformerCaptioningModel`  
**Architecture:** Multi-head attention + FFN  

### 5. `src/train.py`
**Purpose:** Training pipeline for both models  
**Key classes:** `Trainer`  
**Handles:** Training loop, validation, checkpointing  

### 6. `src/evaluate.py`
**Purpose:** Compute evaluation metrics  
**Metrics:** BLEU, METEOR, CIDEr, ROUGE-L  

### 7. `src/visualize_attention.py`
**Purpose:** Generate attention heatmap visualizations  
**Output:** Image with attention overlay  

### 8. `src/demo.py`
**Purpose:** Interactive Gradio web demo  
**Features:** Upload image, generate caption, compare models  

---

## 📝 Training Checklist

Before starting training:

- [ ] Dataset downloaded and extracted to `data/images/`
- [ ] Dataset prepared: `python src/prepare_data.py`
- [ ] PyTorch installed with GPU support (CUDA/MPS)
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] Sufficient disk space (~10GB)
- [ ] GPU memory checked (8GB+ recommended)
- [ ] Training config reviewed in `src/train.py`

During training:

- [ ] Monitor training log: `tail -f outputs/training.log`
- [ ] Check GPU usage: Activity Monitor (Mac) or `nvidia-smi` (NVIDIA)
- [ ] Verify loss is decreasing
- [ ] Checkpoints being saved to `models/`

After training:

- [ ] Evaluate models: `python src/run_evaluation.py`
- [ ] Generate visualizations: `python src/visualize_attention.py`
- [ ] Test demo: `python src/demo.py`
- [ ] Review metrics and compare RNN vs Transformer

---

## 🎓 Academic Requirements

### What You Need for Your Report

1. **Architecture Diagrams**
   - Draw RNN architecture with attention
   - Draw Transformer architecture
   - Show data flow

2. **Training Curves**
   - Plot training loss vs epoch
   - Plot validation loss vs epoch
   - Compare RNN vs Transformer

3. **Evaluation Results**
   - Table of BLEU, METEOR, CIDEr, ROUGE scores
   - Compare with baseline/other papers
   - Statistical significance tests

4. **Attention Visualizations**
   - Show 5-10 examples
   - Highlight interesting cases
   - Explain what model learned

5. **Sample Captions**
   - Show generated vs ground truth
   - Include good and bad examples
   - Analyze failure cases

6. **Ablation Studies** (optional)
   - Effect of attention mechanism
   - Effect of number of layers
   - Effect of embedding dimension

---

## 🔧 Common Issues & Solutions

### Issue: Training is slow

**Solutions:**
- Increase batch size
- Use GPU instead of CPU
- Reduce number of epochs for testing
- Use smaller model

### Issue: Out of memory

**Solutions:**
- Reduce batch size
- Reduce model size (hidden_dim, num_layers)
- Use gradient accumulation
- Clear GPU cache

### Issue: Loss not decreasing

**Solutions:**
- Check learning rate (try 1e-4 to 1e-3)
- Verify data loading is correct
- Check for bugs in model forward pass
- Try different optimizer

### Issue: Poor caption quality

**Solutions:**
- Train for more epochs
- Increase model capacity
- Use beam search instead of greedy decoding
- Try different attention mechanism

---

## 📚 Cursor Commands Quick Reference

```bash
# Ask about specific code
@workspace Explain the attention mechanism in src/rnn_model.py

# Debug errors
@workspace Why am I getting this error: [paste error]

# Modify code
@workspace Add beam search to the caption generation

# Optimize
@workspace How can I make training faster?

# Understand architecture
@workspace What's the difference between RNN and Transformer models?

# Get help with setup
@workspace How do I set up training on M3 Pro?
```

---

## 🎯 Expected Final Results

### Model Performance

| Metric | RNN | Transformer | Winner |
|--------|-----|-------------|--------|
| BLEU-1 | 0.58 | 0.62 | Transformer |
| BLEU-4 | 0.24 | 0.28 | Transformer |
| METEOR | 0.24 | 0.27 | Transformer |
| CIDEr | 0.75 | 0.85 | Transformer |
| ROUGE-L | 0.48 | 0.52 | Transformer |

### Training Time (M3 Pro)

- RNN: 2-2.5 hours
- Transformer: 2.5-3 hours
- Total: ~5 hours

### Model Size

- RNN: ~630 MB
- Transformer: ~730 MB

---

**This document contains everything you need to understand and work with the image captioning project in Cursor AI. Use the prompts above to get help with specific tasks!** 🚀

