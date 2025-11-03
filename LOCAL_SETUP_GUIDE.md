# Local Setup Guide - Image Captioning Project

This guide will help you set up and train the image captioning models on your local computer.

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (Python 3.11 recommended)
- **NVIDIA GPU** (recommended for fast training, but CPU works too)
- **8GB+ RAM** (16GB recommended)
- **10GB+ free disk space**

---

## 📥 Step 1: Clone the Repository

```bash
# Clone from GitHub
git clone https://github.com/yichi0812/image-captioning-neural-networks.git
cd image-captioning-neural-networks
```

---

## 🔧 Step 2: Install Dependencies

### Option A: Using pip (Recommended)

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install PyTorch with CUDA support (for NVIDIA GPU)
# Visit https://pytorch.org/get-started/locally/ to get the right command for your system
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Option B: Using conda

```bash
# Create conda environment
conda create -n image-caption python=3.11
conda activate image-caption

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

---

## 📊 Step 3: Prepare the Dataset

You already have `archive.zip` from Kaggle. Extract it:

```bash
# Create data directory
mkdir -p data

# Extract the dataset
unzip archive.zip -d data/

# Rename Images folder to lowercase (if needed)
mv data/Images data/images

# Prepare train/val/test splits
python src/prepare_data.py --dataset flickr8k --data-dir ./data
```

**Expected output:**
```
Found 8091 images with captions
Dataset split:
  Train: 6472 images
  Val:   809 images
  Test:  810 images
```

---

## 🎯 Step 4: Train the Models

### Train RNN Model

```bash
# Start RNN training (20 epochs, ~1-2 hours on GPU)
python src/train.py --model rnn --epochs 20 --batch-size 32

# Or with custom settings:
python src/train.py \
    --model rnn \
    --epochs 50 \
    --batch-size 64 \
    --learning-rate 0.0003 \
    --data-dir ./data
```

### Train Transformer Model

```bash
# Start Transformer training (20 epochs, ~1-2 hours on GPU)
python src/train.py --model transformer --epochs 20 --batch-size 32

# Or with custom settings:
python src/train.py \
    --model transformer \
    --epochs 50 \
    --batch-size 64 \
    --learning-rate 0.0001 \
    --data-dir ./data
```

### Train Both Models Sequentially

```bash
# This will train both models one after another
python src/train.py
```

---

## ⚡ GPU vs CPU Training Time

| Hardware | RNN (20 epochs) | Transformer (20 epochs) | Total |
|----------|-----------------|-------------------------|-------|
| **NVIDIA RTX 3080** | ~1 hour | ~1.5 hours | ~2.5 hours |
| **NVIDIA GTX 1660** | ~2 hours | ~3 hours | ~5 hours |
| **CPU (8 cores)** | ~15 hours | ~18 hours | ~33 hours |

---

## 📈 Monitor Training Progress

### View Live Training Log

```bash
# Watch training progress in real-time
tail -f outputs/training.log

# Or on Windows:
Get-Content outputs/training.log -Wait
```

### Check Training Status

The training script will print:
- Current epoch and batch progress
- Training and validation loss
- Learning rate
- Time per epoch
- Best validation loss

Example output:
```
Epoch 5/20
--------------------------------------------------
Training: 100%|██████████| 2023/2023 [12:34<00:00,  2.68it/s]
Validation: 100%|██████████| 253/253 [01:23<00:00,  3.04it/s]
Train Loss: 2.1234
Val Loss: 2.0567
Learning Rate: 0.000300
Time: 754.32s
Validation loss improved from 2.1234 to 2.0567
```

---

## 🎮 Step 5: Run the Demo

After training, launch the interactive demo:

```bash
# Launch Gradio demo
python src/demo.py \
    --rnn-checkpoint models/rnn/checkpoint_best.pth \
    --transformer-checkpoint models/transformer/checkpoint_best.pth

# The demo will open in your browser at http://localhost:7860
```

---

## 📊 Step 6: Evaluate the Models

```bash
# Evaluate RNN model
python src/evaluate.py --model rnn --checkpoint models/rnn/checkpoint_best.pth

# Evaluate Transformer model
python src/evaluate.py --model transformer --checkpoint models/transformer/checkpoint_best.pth

# Compare both models
python src/run_evaluation.py
```

**Expected metrics:**
- BLEU-1: 0.55-0.65
- BLEU-4: 0.20-0.30
- METEOR: 0.22-0.28
- CIDEr: 0.70-0.90
- ROUGE-L: 0.45-0.55

---

## 🎨 Step 7: Generate Attention Visualizations

```bash
# Generate attention heatmaps for a specific image
python src/visualize_attention.py \
    --image data/images/1000268201_693b08cb0e.jpg \
    --model rnn \
    --checkpoint models/rnn/checkpoint_best.pth \
    --output outputs/attention_viz.png

# Generate for multiple images
python src/visualize_attention.py \
    --image-dir data/images \
    --num-samples 10 \
    --output-dir outputs/attention_visualizations
```

---

## 🔧 Troubleshooting

### CUDA Out of Memory

If you get CUDA out of memory errors:

```bash
# Reduce batch size
python src/train.py --model rnn --batch-size 16

# Or use gradient accumulation
python src/train.py --model rnn --batch-size 8 --accumulation-steps 4
```

### Slow Training on CPU

If training is too slow on CPU:

```bash
# Reduce dataset size for testing
python src/train.py --model rnn --epochs 5 --max-samples 1000

# Or use fewer epochs
python src/train.py --model rnn --epochs 10
```

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check PyTorch installation
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## 📁 Project Structure After Training

```
image-captioning-neural-networks/
├── data/
│   ├── images/                 # 8,091 images
│   ├── captions.txt           # Original captions
│   ├── train_captions.json    # Training split
│   ├── val_captions.json      # Validation split
│   ├── test_captions.json     # Test split
│   └── vocabulary.pkl         # Built vocabulary
├── models/
│   ├── rnn/
│   │   ├── checkpoint_best.pth      # Best RNN model
│   │   ├── checkpoint_latest.pth    # Latest RNN checkpoint
│   │   ├── checkpoint_final.pth     # Final RNN model
│   │   └── training_history.json    # Training metrics
│   └── transformer/
│       ├── checkpoint_best.pth      # Best Transformer model
│       ├── checkpoint_latest.pth    # Latest checkpoint
│       ├── checkpoint_final.pth     # Final model
│       └── training_history.json    # Training metrics
├── outputs/
│   ├── training.log           # Training logs
│   ├── evaluation_results.json # Evaluation metrics
│   └── attention_visualizations/ # Attention heatmaps
└── src/
    └── (all source code files)
```

---

## 🎯 Quick Commands Reference

```bash
# Setup
git clone https://github.com/yichi0812/image-captioning-neural-networks.git
cd image-captioning-neural-networks
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Prepare data
unzip archive.zip -d data/
mv data/Images data/images
python src/prepare_data.py --dataset flickr8k --data-dir ./data

# Train models
python src/train.py  # Trains both RNN and Transformer

# Evaluate
python src/run_evaluation.py

# Demo
python src/demo.py

# Visualize attention
python src/visualize_attention.py --image data/images/sample.jpg
```

---

## 💡 Tips for Best Results

1. **Use GPU**: Training on GPU is 10-20x faster than CPU
2. **Larger batch size**: Use batch size 64-128 on GPU for faster training
3. **More epochs**: Train for 30-50 epochs for better results
4. **Early stopping**: The script automatically stops if validation loss doesn't improve
5. **Experiment**: Try different learning rates, model sizes, and attention mechanisms

---

## 📝 Training Configuration

Default hyperparameters (can be customized):

### RNN Model
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

### Transformer Model
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

---

## 🎓 For Your School Project

After training locally, you'll have:

1. ✅ **Trained models** with real performance metrics
2. ✅ **Evaluation results** (BLEU, METEOR, CIDEr scores)
3. ✅ **Attention visualizations** for your report
4. ✅ **Training curves** showing learning progress
5. ✅ **Interactive demo** to showcase your work
6. ✅ **Model comparison** (RNN vs Transformer)

All ready for your academic submission!

---

## 🆘 Need Help?

- Check the main README.md for detailed documentation
- Review PROJECT_REPORT.md for implementation details
- Look at USAGE_GUIDE.md for more examples
- Check the GitHub Issues page

---

**Good luck with your training! 🚀**

