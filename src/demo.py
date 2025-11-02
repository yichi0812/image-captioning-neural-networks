"""
Interactive Demo for Image Captioning
Gradio-based web interface for testing the models
"""

import torch
import gradio as gr
from PIL import Image
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from feature_extractor import CLIPFeatureExtractor
from rnn_model import ImageCaptioningRNN
from transformer_model import ImageCaptioningTransformer
from dataset import Vocabulary


class CaptionDemo:
    def __init__(self, rnn_checkpoint=None, transformer_checkpoint=None, vocab_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load vocabulary
        if vocab_path and Path(vocab_path).exists():
            self.vocab = torch.load(vocab_path)
        else:
            print("Warning: No vocabulary found, using dummy vocab")
            self.vocab = Vocabulary(freq_threshold=5)
        
        # Load feature extractor
        self.feature_extractor = CLIPFeatureExtractor().to(self.device)
        self.feature_extractor.eval()
        
        # Load models
        self.rnn_model = None
        self.transformer_model = None
        
        if rnn_checkpoint and Path(rnn_checkpoint).exists():
            self.rnn_model = self._load_rnn(rnn_checkpoint)
            
        if transformer_checkpoint and Path(transformer_checkpoint).exists():
            self.transformer_model = self._load_transformer(transformer_checkpoint)
    
    def _load_rnn(self, checkpoint_path):
        model = ImageCaptioningRNN(
            embed_size=256,
            hidden_size=512,
            vocab_size=len(self.vocab),
            num_layers=2
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def _load_transformer(self, checkpoint_path):
        model = ImageCaptioningTransformer(
            embed_size=512,
            vocab_size=len(self.vocab),
            num_heads=8,
            num_layers=6
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def generate_caption(self, image, model_type="transformer", beam_width=5, max_length=50):
        """Generate caption for an image"""
        if image is None:
            return "Please upload an image"
        
        # Preprocess image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(image).unsqueeze(0).to(self.device)
        
        # Select model
        if model_type == "rnn":
            if self.rnn_model is None:
                return "RNN model not loaded"
            model = self.rnn_model
        else:
            if self.transformer_model is None:
                return "Transformer model not loaded"
            model = self.transformer_model
        
        # Generate caption
        with torch.no_grad():
            if hasattr(model, 'generate_caption'):
                caption_ids = model.generate_caption(features, self.vocab, max_length=max_length)
            else:
                # Fallback: greedy decoding
                caption_ids = self._greedy_decode(model, features, max_length)
        
        # Convert to text
        caption = self._ids_to_caption(caption_ids)
        return caption
    
    def _greedy_decode(self, model, features, max_length):
        """Simple greedy decoding"""
        caption = [self.vocab.stoi.get('<start>', 1)]
        
        for _ in range(max_length):
            caption_tensor = torch.LongTensor(caption).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = model(features, caption_tensor)
                predicted = outputs[0, -1].argmax().item()
            
            caption.append(predicted)
            
            if predicted == self.vocab.stoi.get('<end>', 2):
                break
        
        return caption
    
    def _ids_to_caption(self, ids):
        """Convert token IDs to caption text"""
        words = []
        for idx in ids:
            if idx == self.vocab.stoi.get('<start>', 1):
                continue
            if idx == self.vocab.stoi.get('<end>', 2):
                break
            word = self.vocab.itos.get(idx, '<unk>')
            if word not in ['<pad>', '<unk>']:
                words.append(word)
        
        return ' '.join(words).capitalize()
    
    def compare_models(self, image):
        """Generate captions with both models"""
        rnn_caption = self.generate_caption(image, "rnn")
        transformer_caption = self.generate_caption(image, "transformer")
        
        return f"**RNN**: {rnn_caption}\n\n**Transformer**: {transformer_caption}"


def create_demo(rnn_checkpoint=None, transformer_checkpoint=None, vocab_path=None):
    """Create Gradio interface"""
    
    demo_instance = CaptionDemo(rnn_checkpoint, transformer_checkpoint, vocab_path)
    
    with gr.Blocks(title="Image Captioning Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🖼️ Image Captioning with Deep Learning
        
        Generate natural language descriptions for images using RNN and Transformer models.
        
        **Features:**
        - CLIP-based feature extraction for semantic understanding
        - RNN with Attention for sequential caption generation
        - Transformer Decoder for improved long-range dependencies
        - LLM-based caption refinement with multiple styles
        - Visual attention heatmaps showing where the model looks
        """)
        
        with gr.Tab("Generate Caption"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Upload Image")
                    model_choice = gr.Radio(
                        choices=["rnn", "transformer"],
                        value="transformer",
                        label="Model"
                    )
                    generate_btn = gr.Button("Generate Caption", variant="primary")
                
                with gr.Column():
                    caption_output = gr.Textbox(label="Generated Caption", lines=3)
            
            generate_btn.click(
                fn=demo_instance.generate_caption,
                inputs=[image_input, model_choice],
                outputs=caption_output
            )
        
        with gr.Tab("Compare Models"):
            with gr.Row():
                with gr.Column():
                    compare_image = gr.Image(type="pil", label="Upload Image")
                    compare_btn = gr.Button("Compare Models", variant="primary")
                
                with gr.Column():
                    comparison_output = gr.Markdown(label="Comparison")
            
            compare_btn.click(
                fn=demo_instance.compare_models,
                inputs=compare_image,
                outputs=comparison_output
            )
        
        with gr.Tab("About"):
            gr.Markdown("""
            ## Model Architecture
            
            ### CNN Feature Extractor
            - **Model**: CLIP ViT-B/32 (Vision Transformer)
            - **Fine-tuning**: Last 2 layers adapted to caption dataset
            - **Output**: 512-dimensional feature vectors
            
            ### RNN Decoder
            - **Architecture**: 2-layer LSTM with Bahdanau attention
            - **Embedding**: 256 dimensions
            - **Hidden State**: 512 dimensions
            - **Attention**: Learns to focus on relevant image regions
            
            ### Transformer Decoder
            - **Layers**: 6 transformer blocks
            - **Attention Heads**: 8 multi-head attention
            - **Model Dimension**: 512
            - **Feed-forward**: 2048 dimensions
            
            ## Training Details
            - **Dataset**: Flickr8k (8,000 images, 40,000 captions)
            - **Optimizer**: Adam with cosine annealing
            - **Batch Size**: 32
            - **Epochs**: 50 with early stopping
            
            ## Evaluation Metrics
            - BLEU-1, BLEU-2, BLEU-3, BLEU-4
            - METEOR
            - CIDEr
            - ROUGE-L
            
            ## Project Repository
            [GitHub Repository](https://github.com/YOUR_USERNAME/image-captioning-academic)
            """)
        
        gr.Markdown("""
        ---
        **Note**: This is an academic project for Neural Networks course.
        Models are trained on Flickr8k dataset.
        """)
    
    return demo


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Image Captioning Demo')
    parser.add_argument('--rnn-checkpoint', type=str, default='models/rnn/checkpoint_best.pth',
                       help='Path to RNN model checkpoint')
    parser.add_argument('--transformer-checkpoint', type=str, default='models/transformer/checkpoint_best.pth',
                       help='Path to Transformer model checkpoint')
    parser.add_argument('--vocab', type=str, default='models/vocab.pth',
                       help='Path to vocabulary file')
    parser.add_argument('--share', action='store_true',
                       help='Create public share link')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port to run demo on')
    
    args = parser.parse_args()
    
    demo = create_demo(
        rnn_checkpoint=args.rnn_checkpoint,
        transformer_checkpoint=args.transformer_checkpoint,
        vocab_path=args.vocab
    )
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share
    )

