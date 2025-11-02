"""
Image Feature Extractor using CLIP and ResNet
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

class CLIPFeatureExtractor(nn.Module):
    """
    CLIP-based feature extractor for image captioning
    Uses pre-trained CLIP vision encoder
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32", freeze=True):
        super().__init__()
        self.model_name = model_name
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        
        # Freeze CLIP weights if specified
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.eval()
        
        # Get feature dimension
        self.feature_dim = self.model.config.vision_config.hidden_size  # 768 for base
        
    def forward(self, pixel_values):
        """
        Extract features from images
        Args:
            pixel_values: Preprocessed image tensors from CLIP processor
        Returns:
            features: (batch_size, feature_dim)
            spatial_features: (batch_size, num_patches, feature_dim) for attention
        """
        with torch.no_grad():
            # Get vision features
            vision_outputs = self.model.vision_model(pixel_values=pixel_values)
            
            # Pooled features (CLS token)
            pooled_features = vision_outputs[1]  # pooler_output
            
            # Spatial features (all patch tokens)
            spatial_features = vision_outputs[0]  # last_hidden_state
            
        return pooled_features, spatial_features
    
    def preprocess(self, image_path):
        """Preprocess image for CLIP"""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs['pixel_values']


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-based feature extractor
    Alternative to CLIP for comparison
    """
    def __init__(self, model_name='resnet50', freeze=True):
        super().__init__()
        
        # Load pre-trained ResNet
        if model_name == 'resnet50':
            resnet = models.resnet50(pretrained=True)
        elif model_name == 'resnet101':
            resnet = models.resnet101(pretrained=True)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Remove final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
        
        self.features.eval()
        self.feature_dim = 2048  # ResNet50/101 output channels
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, images):
        """
        Extract features from images
        Args:
            images: (batch, 3, 224, 224) tensor
        Returns:
            pooled_features: (batch, 2048)
            spatial_features: (batch, 49, 2048) for 7x7 feature map
        """
        with torch.no_grad():
            # Extract spatial features
            spatial = self.features(images)  # (batch, 2048, 7, 7)
            
            # Reshape for attention: (batch, 49, 2048)
            batch_size = spatial.size(0)
            spatial_features = spatial.view(batch_size, self.feature_dim, -1)  # (batch, 2048, 49)
            spatial_features = spatial_features.permute(0, 2, 1)  # (batch, 49, 2048)
            
            # Pooled features
            pooled = self.avgpool(spatial)  # (batch, 2048, 1, 1)
            pooled_features = pooled.view(batch_size, -1)  # (batch, 2048)
            
        return pooled_features, spatial_features
    
    def preprocess(self, image_path):
        """Preprocess image for ResNet"""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        return self.transform(image).unsqueeze(0)


def get_feature_extractor(model_type='clip', **kwargs):
    """
    Factory function to get feature extractor
    
    Args:
        model_type: 'clip' or 'resnet'
        **kwargs: Additional arguments for the extractor
    
    Returns:
        Feature extractor instance
    """
    if model_type == 'clip':
        return CLIPFeatureExtractor(**kwargs)
    elif model_type == 'resnet':
        return ResNetFeatureExtractor(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test feature extractors
    print("Testing CLIP Feature Extractor...")
    clip_extractor = CLIPFeatureExtractor()
    print(f"CLIP feature dimension: {clip_extractor.feature_dim}")
    
    print("\nTesting ResNet Feature Extractor...")
    resnet_extractor = ResNetFeatureExtractor()
    print(f"ResNet feature dimension: {resnet_extractor.feature_dim}")
    
    # Test with dummy image
    dummy_image = Image.new('RGB', (224, 224), color='red')
    
    print("\nTesting CLIP preprocessing and forward pass...")
    clip_input = clip_extractor.preprocess(dummy_image)
    pooled, spatial = clip_extractor(clip_input)
    print(f"CLIP pooled features shape: {pooled.shape}")
    print(f"CLIP spatial features shape: {spatial.shape}")
    
    print("\nTesting ResNet preprocessing and forward pass...")
    resnet_input = resnet_extractor.preprocess(dummy_image)
    pooled, spatial = resnet_extractor(resnet_input)
    print(f"ResNet pooled features shape: {pooled.shape}")
    print(f"ResNet spatial features shape: {spatial.shape}")
    
    print("\n✓ Feature extractors working correctly!")

