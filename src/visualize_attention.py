"""
Visualize attention weights as heatmaps
"""
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import cv2
from pathlib import Path


class AttentionVisualizer:
    """
    Visualize attention weights on images
    """
    def __init__(self):
        pass
    
    def visualize_attention(self, image_path, caption, attention_weights, 
                          output_path=None, grid_size=(7, 7)):
        """
        Visualize attention weights as heatmaps overlaid on image
        
        Args:
            image_path: Path to input image
            caption: Generated caption (string or list of words)
            attention_weights: Attention weights (seq_len, num_pixels) or (seq_len, H, W)
            output_path: Path to save visualization
            grid_size: Size of attention grid (H, W)
        """
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        img_array = np.array(image)
        img_h, img_w = img_array.shape[:2]
        
        # Parse caption
        if isinstance(caption, str):
            words = caption.split()
        else:
            words = caption
        
        # Convert attention weights to numpy
        if torch.is_tensor(attention_weights):
            attention_weights = attention_weights.cpu().numpy()
        
        seq_len = attention_weights.shape[0]
        num_pixels = attention_weights.shape[1]
        
        # Determine grid size
        if len(attention_weights.shape) == 2:
            # Reshape to grid
            grid_h, grid_w = grid_size
            if num_pixels == grid_h * grid_w:
                attention_grid = attention_weights.reshape(seq_len, grid_h, grid_w)
            else:
                # Try to infer grid size
                grid_size = int(np.sqrt(num_pixels))
                attention_grid = attention_weights.reshape(seq_len, grid_size, grid_size)
        else:
            attention_grid = attention_weights
        
        # Create figure
        num_words = min(len(words), seq_len)
        cols = min(4, num_words)
        rows = (num_words + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot attention for each word
        for idx in range(num_words):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            # Get attention map for this word
            attn_map = attention_grid[idx]
            
            # Resize attention map to image size
            attn_resized = cv2.resize(attn_map, (img_w, img_h))
            
            # Normalize
            attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
            
            # Display image
            ax.imshow(img_array)
            
            # Overlay attention heatmap
            ax.imshow(attn_resized, alpha=0.6, cmap='jet')
            
            # Set title
            word = words[idx] if idx < len(words) else ''
            ax.set_title(f'"{word}"', fontsize=12, fontweight='bold')
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(num_words, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Attention visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_attention_gif(self, image_path, caption, attention_weights, 
                           output_path, grid_size=(7, 7), duration=500):
        """
        Create animated GIF showing attention progression
        
        Args:
            image_path: Path to input image
            caption: Generated caption
            attention_weights: Attention weights
            output_path: Path to save GIF
            grid_size: Size of attention grid
            duration: Duration per frame in ms
        """
        from PIL import Image, ImageDraw, ImageFont
        
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        img_array = np.array(image)
        img_h, img_w = img_array.shape[:2]
        
        # Parse caption
        if isinstance(caption, str):
            words = caption.split()
        else:
            words = caption
        
        # Convert attention weights
        if torch.is_tensor(attention_weights):
            attention_weights = attention_weights.cpu().numpy()
        
        seq_len = attention_weights.shape[0]
        num_pixels = attention_weights.shape[1]
        
        # Reshape attention
        if len(attention_weights.shape) == 2:
            grid_h, grid_w = grid_size
            if num_pixels == grid_h * grid_w:
                attention_grid = attention_weights.reshape(seq_len, grid_h, grid_w)
            else:
                grid_size = int(np.sqrt(num_pixels))
                attention_grid = attention_weights.reshape(seq_len, grid_size, grid_size)
        else:
            attention_grid = attention_weights
        
        # Create frames
        frames = []
        for idx in range(min(len(words), seq_len)):
            # Get attention map
            attn_map = attention_grid[idx]
            attn_resized = cv2.resize(attn_map, (img_w, img_h))
            attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
            
            # Create heatmap
            heatmap = cm.jet(attn_resized)[:, :, :3]
            heatmap = (heatmap * 255).astype(np.uint8)
            
            # Blend with original image
            blended = cv2.addWeighted(img_array, 0.5, heatmap, 0.5, 0)
            
            # Convert to PIL
            frame = Image.fromarray(blended)
            
            # Add text
            draw = ImageDraw.Draw(frame)
            text = ' '.join(words[:idx+1])
            
            # Draw text with background
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Draw background rectangle
            padding = 10
            draw.rectangle(
                [(10, img_h - text_height - 2*padding), 
                 (text_width + 2*padding, img_h - padding)],
                fill=(0, 0, 0, 180)
            )
            
            # Draw text
            draw.text((padding, img_h - text_height - padding), text, 
                     fill=(255, 255, 255), font=font)
            
            frames.append(frame)
        
        # Save GIF
        if frames:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0
            )
            print(f"Attention GIF saved to {output_path}")


def visualize_model_attention(model, image_path, vocab, output_dir, device='cpu'):
    """
    Generate attention visualizations for a model
    
    Args:
        model: Trained captioning model
        image_path: Path to image
        vocab: Vocabulary
        output_dir: Directory to save visualizations
        device: Device to run on
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    model.eval()
    model = model.to(device)
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    img_tensor = model.feature_extractor.preprocess(image).to(device)
    
    # Generate caption with attention
    with torch.no_grad():
        caption, attention_weights = model.generate_caption(img_tensor, vocab)
    
    print(f"Generated caption: {caption}")
    print(f"Attention shape: {attention_weights.shape if attention_weights is not None else 'None'}")
    
    if attention_weights is not None:
        # Create visualizer
        visualizer = AttentionVisualizer()
        
        # Determine grid size based on feature extractor
        if hasattr(model.feature_extractor, 'feature_dim'):
            if model.feature_extractor.feature_dim == 768:  # CLIP
                grid_size = (7, 7)  # Approximate for 50 patches
            elif model.feature_extractor.feature_dim == 2048:  # ResNet
                grid_size = (7, 7)
            else:
                grid_size = (7, 7)
        else:
            grid_size = (7, 7)
        
        # Generate static visualization
        output_path = output_dir / f"{Path(image_path).stem}_attention.png"
        visualizer.visualize_attention(
            image_path, caption, attention_weights,
            output_path=output_path,
            grid_size=grid_size
        )
        
        # Generate GIF
        gif_path = output_dir / f"{Path(image_path).stem}_attention.gif"
        visualizer.create_attention_gif(
            image_path, caption, attention_weights,
            output_path=gif_path,
            grid_size=grid_size
        )
    
    return caption, attention_weights


if __name__ == "__main__":
    print("Attention visualization module loaded successfully!")

