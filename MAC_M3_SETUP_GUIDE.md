# Mac M3 Pro Setup Guide - Image Captioning Project

Complete guide for training image captioning models on **Apple Silicon M3 Pro** with GPU acceleration.

---

## 🍎 Why M3 Pro is Great for This Project

- ✅ **Metal Performance Shaders (MPS)** - Native GPU acceleration in PyTorch
- ✅ **Unified Memory** - Efficient memory sharing between CPU and GPU
- ✅ **18GB+ RAM** - Plenty for large batch sizes
- ✅ **Neural Engine** - Hardware ML acceleration
- ✅ **Energy Efficient** - Train for hours without overheating

**Expected Training Time:** 3-5 hours for both models (RNN + Transformer, 20 epochs each)

---

## 🚀 Quick Start for M3 Pro

### Step 1: Install Homebrew (if not already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 2: Install Python 3.11

```bash
# Install Python via Homebrew
brew install python@3.11

# Verify installation
python3.11 --version
```

### Step 3: Clone the Repository

```bash
cd ~/Documents  # or wherever you want to work
git clone https://github.com/yichi0812/image-captioning-neural-networks.git
cd image-captioning-neural-networks
```

### Step 4: Create Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate it
source venv/bin/activate

# Your terminal should now show (venv) at the beginning
```

### Step 5: Install PyTorch with MPS Support

```bash
# Install PyTorch optimized for Apple Silicon
pip install --upgrade pip
pip install torch torchvision torchaudio

# Verify MPS (Metal) is available
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
# Should print: MPS available: True
```

### Step 6: Install Other Dependencies

```bash
pip install transformers pillow tqdm nltk matplotlib seaborn gradio
pip install pycocoevalcap
```

### Step 7: Prepare the Dataset

```bash
# Create data directory
mkdir -p data

# Copy your archive.zip to the project folder, then:
unzip archive.zip -d data/

# Rename Images folder
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
✓ Dataset preparation complete!
```

---

## ⚡ Step 8: Train with MPS Acceleration

### Update train.py for MPS

The code will automatically detect and use MPS. Verify by checking the device:

```bash
# Check if MPS will be used
python3 -c "import torch; print('Device:', 'mps' if torch.backends.mps.is_available() else 'cpu')"
```

### Train Both Models

```bash
# Train RNN model (will use MPS automatically)
python src/train.py --model rnn --epochs 20 --batch-size 64

# Then train Transformer model
python src/train.py --model transformer --epochs 20 --batch-size 64

# Or train both sequentially
python src/train.py
```

### Optimize for M3 Pro

```bash
# Use larger batch size (M3 Pro has plenty of memory)
python src/train.py --batch-size 128

# Enable mixed precision for faster training
# (Add this to your training script if not already there)
```

---

## 📊 Expected Performance on M3 Pro

| Model | Epochs | Batch Size | Time | Memory Usage |
|-------|--------|------------|------|--------------|
| RNN | 20 | 64 | ~2-2.5 hours | ~8GB |
| Transformer | 20 | 64 | ~2.5-3 hours | ~10GB |
| **Total** | **40** | **64** | **~5 hours** | **~10GB peak** |

With batch size 128:
- **RNN**: ~1.5 hours
- **Transformer**: ~2 hours
- **Total**: ~3.5 hours

---

## 🎯 Monitor Training

### Open a new terminal and watch progress:

```bash
# Navigate to project
cd ~/Documents/image-captioning-neural-networks

# Watch training log
tail -f outputs/training.log
```

### Check GPU usage:

```bash
# Monitor system resources
sudo powermetrics --samplers gpu_power -i1000 -n1

# Or use Activity Monitor app:
# Open Activity Monitor > Window > GPU History
```

---

## 🔧 M3 Pro Specific Optimizations

### 1. Increase Batch Size

M3 Pro has 18GB unified memory - use it!

```bash
# Try batch size 128 or even 256
python src/train.py --batch-size 128
```

### 2. Enable MPS Fallback

Add this to the top of `src/train.py` if you encounter issues:

```python
import torch
import os

# Enable MPS fallback for unsupported operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")
```

### 3. Optimize Data Loading

```python
# In your DataLoader, use more workers
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=8,  # M3 Pro has 12 cores, use 8 for data loading
    pin_memory=False  # Not needed for MPS
)
```

---

## 🎮 Run the Demo

After training:

```bash
# Launch interactive demo
python src/demo.py \
    --rnn-checkpoint models/rnn/checkpoint_best.pth \
    --transformer-checkpoint models/transformer/checkpoint_best.pth

# Opens at http://localhost:7860
```

---

## 📊 Evaluate Models

```bash
# Evaluate RNN
python src/evaluate.py --model rnn --checkpoint models/rnn/checkpoint_best.pth

# Evaluate Transformer
python src/evaluate.py --model transformer --checkpoint models/transformer/checkpoint_best.pth

# Compare both
python src/run_evaluation.py
```

---

## 🎨 Generate Visualizations

```bash
# Create attention heatmaps
python src/visualize_attention.py \
    --image data/images/1000268201_693b08cb0e.jpg \
    --model rnn \
    --checkpoint models/rnn/checkpoint_best.pth \
    --output outputs/attention_viz.png
```

---

## 🐛 Troubleshooting M3 Pro Issues

### Issue: "MPS backend not available"

**Solution:**
```bash
# Update PyTorch to latest version
pip install --upgrade torch torchvision torchaudio

# Verify macOS version (need macOS 12.3+)
sw_vers
```

### Issue: "Operation not supported on MPS"

**Solution:**
```bash
# Enable fallback to CPU for unsupported ops
export PYTORCH_ENABLE_MPS_FALLBACK=1
python src/train.py
```

Or add to your script:
```python
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
```

### Issue: Training is slow

**Checklist:**
- ✅ Verify MPS is being used: `python3 -c "import torch; print(torch.backends.mps.is_available())"`
- ✅ Increase batch size: `--batch-size 128`
- ✅ Close other apps to free up memory
- ✅ Ensure Mac is plugged in (not on battery)
- ✅ Check Activity Monitor for GPU usage

### Issue: Out of memory

**Solution:**
```bash
# Reduce batch size
python src/train.py --batch-size 32

# Or use gradient accumulation
python src/train.py --batch-size 16 --accumulation-steps 4
```

---

## 💡 M3 Pro Best Practices

### 1. Keep Your Mac Cool
- Use in a well-ventilated area
- Consider a laptop stand for better airflow
- Training will make the fans spin up - this is normal

### 2. Power Settings
- Keep Mac plugged in during training
- Disable sleep mode:
  ```bash
  # Prevent sleep while training
  caffeinate -i python src/train.py
  ```

### 3. Monitor Resources
```bash
# Check memory usage
top -o mem

# Check CPU usage
top -o cpu

# Monitor GPU
sudo powermetrics --samplers gpu_power
```

### 4. Background Training
```bash
# Run training in background
nohup python src/train.py > outputs/training.log 2>&1 &

# Check progress
tail -f outputs/training.log

# Check if still running
ps aux | grep train.py
```

---

## 📈 Performance Comparison

| Device | RNN + Transformer (20 epochs each) |
|--------|-----------------------------------|
| **M3 Pro** | ~3-5 hours |
| **M2 Pro** | ~4-6 hours |
| **M1 Pro** | ~5-7 hours |
| **Intel Mac** | ~30+ hours (CPU only) |
| **NVIDIA RTX 3080** | ~2.5 hours |
| **NVIDIA GTX 1660** | ~5 hours |

---

## 🎓 Complete Training Workflow

```bash
# 1. Setup (one-time)
git clone https://github.com/yichi0812/image-captioning-neural-networks.git
cd image-captioning-neural-networks
python3.11 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio transformers pillow tqdm nltk matplotlib seaborn gradio pycocoevalcap

# 2. Prepare data (one-time)
unzip archive.zip -d data/
mv data/Images data/images
python src/prepare_data.py --dataset flickr8k --data-dir ./data

# 3. Train models (~3-5 hours)
caffeinate -i python src/train.py

# 4. Evaluate
python src/run_evaluation.py

# 5. Demo
python src/demo.py

# 6. Visualize
python src/visualize_attention.py --image data/images/sample.jpg
```

---

## 🔥 Pro Tips for M3 Pro

1. **Use batch size 128** - M3 Pro can handle it
2. **Train overnight** - Let it run while you sleep
3. **Monitor with Activity Monitor** - Watch GPU usage
4. **Use caffeinate** - Prevent sleep during training
5. **Close other apps** - Free up memory for training
6. **Keep plugged in** - Don't train on battery
7. **Enable MPS fallback** - For compatibility
8. **Use num_workers=8** - Leverage the 12-core CPU

---

## 📝 Quick Commands

```bash
# Verify MPS
python3 -c "import torch; print('MPS:', torch.backends.mps.is_available())"

# Train with optimal settings for M3 Pro
caffeinate -i python src/train.py --batch-size 128

# Monitor training
tail -f outputs/training.log

# Check GPU usage
sudo powermetrics --samplers gpu_power -i1000 -n1

# Run demo
python src/demo.py
```

---

## ✅ Checklist

Before starting training:
- [ ] Python 3.11 installed
- [ ] Virtual environment created and activated
- [ ] PyTorch with MPS support installed
- [ ] MPS available (check with Python command)
- [ ] Dataset extracted and prepared
- [ ] Mac plugged into power
- [ ] Other apps closed
- [ ] Enough disk space (~10GB)

---

## 🆘 Need Help?

Common issues and solutions:
1. **MPS not available** → Update PyTorch and macOS
2. **Slow training** → Increase batch size, close other apps
3. **Out of memory** → Reduce batch size
4. **Mac sleeping** → Use `caffeinate` command
5. **Import errors** → Reinstall dependencies

---

**Your M3 Pro is perfect for this project! Training should take 3-5 hours instead of 30+ hours on CPU. Enjoy! 🚀**

