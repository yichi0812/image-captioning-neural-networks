# Usage Guide - Image Captioning Project

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yichi0812/image-captioning-neural-networks.git
cd image-captioning-neural-networks
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (required for evaluation)
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

### 3. Prepare Dataset

#### Option A: Flickr8k (Recommended for Academic Projects)

```bash
# Download from Kaggle
# Visit: https://www.kaggle.com/datasets/adityajn105/flickr8k
# Download and extract to data/

# Prepare dataset
python src/prepare_data.py --dataset flickr8k --data-dir ./data
```

#### Option B: MS COCO (For Advanced Experiments)

```bash
# Download from http://cocodataset.org/#download
# Extract train2017, val2017, and annotations to data/

python src/prepare_data.py --dataset coco --data-dir ./data
```

### 4. Train Models

#### Train RNN Model

```bash
python src/train.py \
    --model rnn \
    --data-dir ./data \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --save-dir ./models/rnn
```

**Expected Output:**
```
Epoch 1/50: Loss=4.2341, Val Loss=3.8765
Epoch 2/50: Loss=3.5432, Val Loss=3.2109
...
Best model saved at epoch 35 with Val Loss=1.2345
```

**Training Time**: ~2-3 hours on GPU, ~12-15 hours on CPU

#### Train Transformer Model

```bash
python src/train.py \
    --model transformer \
    --data-dir ./data \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --save-dir ./models/transformer
```

**Training Time**: ~3-4 hours on GPU, ~15-20 hours on CPU

### 5. Evaluate Models

```bash
# Evaluate RNN model
python src/evaluate.py \
    --model rnn \
    --checkpoint ./models/rnn/checkpoint_best.pth \
    --data-dir ./data \
    --split test

# Evaluate Transformer model
python src/evaluate.py \
    --model transformer \
    --checkpoint ./models/transformer/checkpoint_best.pth \
    --data-dir ./data \
    --split test

# Compare both models
python src/run_evaluation.py --compare-models
```

**Output:**
```
Evaluation Results:
------------------
Model: RNN
BLEU-1: 0.6543
BLEU-4: 0.2341
METEOR: 0.2567
CIDEr: 0.7654
ROUGE-L: 0.4321

Model: Transformer
BLEU-1: 0.7123
BLEU-4: 0.2987
METEOR: 0.2891
CIDEr: 0.8765
ROUGE-L: 0.4876
```

### 6. Generate Captions for New Images

```bash
# Single image
python src/generate_caption.py \
    --image path/to/image.jpg \
    --model transformer \
    --checkpoint ./models/transformer/checkpoint_best.pth

# Batch processing
python src/generate_caption.py \
    --image-dir path/to/images/ \
    --model transformer \
    --checkpoint ./models/transformer/checkpoint_best.pth \
    --output captions.json
```

### 7. Visualize Attention

```bash
# Generate attention heatmap
python src/visualize_attention.py \
    --image path/to/image.jpg \
    --model rnn \
    --checkpoint ./models/rnn/checkpoint_best.pth \
    --output attention_heatmap.png

# Compare RNN vs Transformer attention
python src/visualize_attention.py \
    --image path/to/image.jpg \
    --compare-models \
    --rnn-checkpoint ./models/rnn/checkpoint_best.pth \
    --transformer-checkpoint ./models/transformer/checkpoint_best.pth
```

### 8. Launch Interactive Demo

```bash
python src/demo.py \
    --rnn-checkpoint ./models/rnn/checkpoint_best.pth \
    --transformer-checkpoint ./models/transformer/checkpoint_best.pth \
    --share  # Optional: create public link
```

Access the demo at: `http://localhost:7860`

---

## Advanced Usage

### Hyperparameter Tuning

```bash
# Experiment with different learning rates
python src/train.py --model transformer --learning-rate 5e-5
python src/train.py --model transformer --learning-rate 1e-4
python src/train.py --model transformer --learning-rate 5e-4

# Experiment with batch sizes
python src/train.py --model transformer --batch-size 16
python src/train.py --model transformer --batch-size 32
python src/train.py --model transformer --batch-size 64

# Experiment with model dimensions
python src/train.py --model transformer --embed-size 256
python src/train.py --model transformer --embed-size 512
python src/train.py --model transformer --embed-size 1024
```

### Caption Refinement with LLM

```bash
# Refine captions using GPT
export OPENAI_API_KEY="your-api-key-here"

python src/caption_refiner.py \
    --input captions.json \
    --output refined_captions.json \
    --style natural  # Options: natural, detailed, concise, poetic
```

### Resume Training

```bash
# Resume from checkpoint
python src/train.py \
    --model transformer \
    --resume ./models/transformer/checkpoint_epoch_20.pth \
    --epochs 50
```

### Export Model for Deployment

```bash
# Convert to TorchScript
python src/export_model.py \
    --checkpoint ./models/transformer/checkpoint_best.pth \
    --output model.pt

# Convert to ONNX
python src/export_model.py \
    --checkpoint ./models/transformer/checkpoint_best.pth \
    --format onnx \
    --output model.onnx
```

---

## Troubleshooting

### Out of Memory Errors

**Problem**: CUDA out of memory during training

**Solutions**:
```bash
# Reduce batch size
python src/train.py --batch-size 16

# Use gradient accumulation
python src/train.py --batch-size 16 --accumulation-steps 2

# Use mixed precision training
python src/train.py --mixed-precision
```

### Slow Training

**Problem**: Training is very slow

**Solutions**:
- Use GPU if available
- Reduce image resolution
- Use fewer data augmentations
- Reduce model size

```bash
# Use smaller model
python src/train.py --embed-size 256 --hidden-size 512

# Reduce image size
python src/train.py --image-size 224
```

### Poor Caption Quality

**Problem**: Generated captions are not good

**Solutions**:
- Train for more epochs
- Increase model capacity
- Use beam search during inference
- Apply caption refinement

```bash
# Use beam search
python src/generate_caption.py --beam-width 5

# Increase model size
python src/train.py --embed-size 512 --hidden-size 1024

# More training
python src/train.py --epochs 100
```

---

## Project Structure Explained

```
image-captioning-neural-networks/
│
├── src/                        # Source code
│   ├── dataset.py             # Data loading and preprocessing
│   ├── feature_extractor.py   # CLIP-based CNN
│   ├── rnn_model.py           # RNN with attention
│   ├── transformer_model.py   # Transformer decoder
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation metrics
│   ├── visualize_attention.py # Attention visualization
│   ├── caption_refiner.py     # LLM refinement
│   └── demo.py                # Interactive demo
│
├── data/                       # Dataset storage
│   ├── Images/                # Image files
│   ├── captions.txt           # Caption annotations
│   └── *_captions.json        # Train/val/test splits
│
├── models/                     # Saved checkpoints
│   ├── rnn/                   # RNN model checkpoints
│   ├── transformer/           # Transformer checkpoints
│   └── vocab.pth              # Vocabulary file
│
├── outputs/                    # Generated outputs
│   ├── training_log.txt       # Training logs
│   ├── evaluation_results.json # Evaluation metrics
│   └── attention_maps/        # Attention visualizations
│
├── docs/                       # Documentation
│   ├── PROJECT_REPORT.md      # Comprehensive report
│   └── USAGE_GUIDE.md         # This file
│
├── notebooks/                  # Jupyter notebooks
│   └── analysis.ipynb         # Data analysis
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview
└── LICENSE                     # MIT License
```

---

## Tips for Academic Submission

### 1. Document Your Experiments

Keep a log of all experiments:
```bash
# Create experiment log
echo "Experiment 1: Baseline RNN" >> experiments.log
python src/train.py --model rnn | tee -a experiments.log

echo "Experiment 2: Transformer" >> experiments.log
python src/train.py --model transformer | tee -a experiments.log
```

### 2. Generate Visualizations

Create figures for your report:
```bash
# Training curves
python src/plot_training.py --log training_log.txt --output training_curves.png

# Attention heatmaps
python src/visualize_attention.py --batch --output-dir attention_maps/

# Model comparison
python src/compare_models.py --output comparison_chart.png
```

### 3. Prepare Presentation

```bash
# Generate sample captions for presentation
python src/generate_samples.py --num-samples 20 --output samples.html

# Create demo video
python src/record_demo.py --output demo.mp4
```

### 4. Write Report

Use the provided template:
- `docs/PROJECT_REPORT.md` - Main report
- Fill in results tables
- Add generated visualizations
- Include code snippets

---

## Common Commands Cheat Sheet

```bash
# Setup
pip install -r requirements.txt
python src/prepare_data.py --dataset flickr8k

# Training
python src/train.py --model rnn --epochs 50
python src/train.py --model transformer --epochs 50

# Evaluation
python src/evaluate.py --model rnn --checkpoint models/rnn/checkpoint_best.pth
python src/run_evaluation.py --compare-models

# Inference
python src/generate_caption.py --image test.jpg --model transformer
python src/visualize_attention.py --image test.jpg --model rnn

# Demo
python src/demo.py --share
```

---

## Additional Resources

- **Dataset**: [Flickr8k on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- **CLIP Paper**: [Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)
- **Attention Paper**: [Show, Attend and Tell](https://arxiv.org/abs/1502.03044)
- **Transformer Paper**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the project report
3. Open an issue on GitHub
4. Contact course instructors

---

**Good luck with your project!** 🚀

