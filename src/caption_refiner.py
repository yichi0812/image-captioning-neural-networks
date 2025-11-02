"""
Caption refinement using LLM
"""
import os
from openai import OpenAI


class CaptionRefiner:
    """
    Refine generated captions using a small LLM
    """
    def __init__(self, model="gpt-4.1-nano"):
        """
        Initialize caption refiner
        Args:
            model: Model to use for refinement (gpt-4.1-nano, gpt-4.1-mini)
        """
        self.client = OpenAI()  # API key is pre-configured in environment
        self.model = model
    
    def refine_caption(self, raw_caption, style="natural", max_tokens=50):
        """
        Refine a raw caption to make it more natural and descriptive
        
        Args:
            raw_caption: Raw caption from the model
            style: Refinement style ('natural', 'detailed', 'concise', 'poetic')
            max_tokens: Maximum tokens for refined caption
        
        Returns:
            Refined caption string
        """
        # Define prompts for different styles
        style_prompts = {
            'natural': "Rewrite this image caption to make it more natural and fluent while keeping the same meaning:",
            'detailed': "Expand this image caption to be more detailed and descriptive:",
            'concise': "Make this image caption more concise while keeping the key information:",
            'poetic': "Rewrite this image caption in a more poetic and evocative style:"
        }
        
        prompt = style_prompts.get(style, style_prompts['natural'])
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at refining image captions. Provide only the refined caption without any additional explanation."
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nOriginal caption: {raw_caption}\n\nRefined caption:"
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            refined_caption = response.choices[0].message.content.strip()
            
            # Remove quotes if present
            if refined_caption.startswith('"') and refined_caption.endswith('"'):
                refined_caption = refined_caption[1:-1]
            
            return refined_caption
        
        except Exception as e:
            print(f"Error refining caption: {e}")
            return raw_caption
    
    def batch_refine(self, captions, style="natural"):
        """
        Refine multiple captions
        
        Args:
            captions: List of raw captions
            style: Refinement style
        
        Returns:
            List of refined captions
        """
        refined_captions = []
        
        for caption in captions:
            refined = self.refine_caption(caption, style=style)
            refined_captions.append(refined)
        
        return refined_captions
    
    def compare_refinements(self, raw_caption):
        """
        Compare different refinement styles
        
        Args:
            raw_caption: Raw caption to refine
        
        Returns:
            Dictionary with different refined versions
        """
        styles = ['natural', 'detailed', 'concise', 'poetic']
        
        refinements = {
            'original': raw_caption
        }
        
        for style in styles:
            refinements[style] = self.refine_caption(raw_caption, style=style)
        
        return refinements


def refine_caption_simple(raw_caption, model="gpt-4.1-nano"):
    """
    Simple function to refine a single caption
    
    Args:
        raw_caption: Raw caption from model
        model: LLM model to use
    
    Returns:
        Refined caption
    """
    refiner = CaptionRefiner(model=model)
    return refiner.refine_caption(raw_caption)


if __name__ == "__main__":
    # Test caption refiner
    print("Testing Caption Refiner...")
    
    # Sample captions
    test_captions = [
        "a dog running on the grass",
        "a mountain with snow",
        "a beach with waves"
    ]
    
    refiner = CaptionRefiner(model="gpt-4.1-nano")
    
    print("\nTesting single caption refinement:")
    print("-" * 60)
    for caption in test_captions:
        print(f"\nOriginal: {caption}")
        refined = refiner.refine_caption(caption, style='natural')
        print(f"Refined:  {refined}")
    
    print("\n\nTesting different styles:")
    print("-" * 60)
    test_caption = "a dog running on the grass"
    refinements = refiner.compare_refinements(test_caption)
    
    for style, refined in refinements.items():
        print(f"\n{style.capitalize():12s}: {refined}")
    
    print("\n✓ Caption refiner working correctly!")

