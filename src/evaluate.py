"""
Evaluation metrics for image captioning
"""
import torch
from pathlib import Path
import json
from tqdm import tqdm
import pickle
from collections import defaultdict
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class CaptionEvaluator:
    """
    Evaluate image captioning models using multiple metrics
    """
    def __init__(self):
        self.smoothing = SmoothingFunction()
    
    def compute_bleu(self, references, hypotheses, max_n=4):
        """
        Compute BLEU scores
        Args:
            references: List of reference captions (list of lists of tokens)
            hypotheses: List of generated captions (list of tokens)
            max_n: Maximum n-gram order
        Returns:
            Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
        """
        bleu_scores = {}
        
        for n in range(1, max_n + 1):
            weights = tuple([1.0/n] * n + [0] * (max_n - n))
            
            # Corpus BLEU
            corpus_score = corpus_bleu(
                references, hypotheses,
                weights=weights,
                smoothing_function=self.smoothing.method1
            )
            bleu_scores[f'BLEU-{n}'] = corpus_score
        
        return bleu_scores
    
    def compute_meteor(self, references, hypotheses):
        """
        Compute METEOR score
        Args:
            references: List of reference captions (list of strings)
            hypotheses: List of generated captions (strings)
        Returns:
            Average METEOR score
        """
        scores = []
        for ref_list, hyp in zip(references, hypotheses):
            # METEOR expects single reference string
            # Use first reference or average over all
            ref_scores = []
            for ref in ref_list:
                try:
                    score = meteor_score([ref], hyp)
                    ref_scores.append(score)
                except:
                    ref_scores.append(0.0)
            
            scores.append(max(ref_scores) if ref_scores else 0.0)
        
        return np.mean(scores)
    
    def compute_cider(self, references, hypotheses):
        """
        Simplified CIDEr-like score based on TF-IDF
        (Full CIDEr requires pycocoevalcap which may have dependencies)
        """
        from collections import Counter
        import math
        
        # Compute document frequency
        df = Counter()
        for ref_list in references:
            unique_ngrams = set()
            for ref in ref_list:
                tokens = ref.split()
                for i in range(len(tokens)):
                    for j in range(i+1, min(i+5, len(tokens)+1)):
                        unique_ngrams.add(' '.join(tokens[i:j]))
            df.update(unique_ngrams)
        
        num_docs = len(references)
        
        # Compute CIDEr for each hypothesis
        scores = []
        for ref_list, hyp in zip(references, hypotheses):
            hyp_tokens = hyp.split()
            hyp_ngrams = Counter()
            for i in range(len(hyp_tokens)):
                for j in range(i+1, min(i+5, len(hyp_tokens)+1)):
                    hyp_ngrams[' '.join(hyp_tokens[i:j])] += 1
            
            # Compute TF-IDF for hypothesis
            hyp_tfidf = {}
            for ngram, count in hyp_ngrams.items():
                tf = count / len(hyp_tokens) if len(hyp_tokens) > 0 else 0
                idf = math.log(num_docs / (df[ngram] + 1))
                hyp_tfidf[ngram] = tf * idf
            
            # Compute average similarity with references
            ref_scores = []
            for ref in ref_list:
                ref_tokens = ref.split()
                ref_ngrams = Counter()
                for i in range(len(ref_tokens)):
                    for j in range(i+1, min(i+5, len(ref_tokens)+1)):
                        ref_ngrams[' '.join(ref_tokens[i:j])] += 1
                
                # Compute TF-IDF for reference
                ref_tfidf = {}
                for ngram, count in ref_ngrams.items():
                    tf = count / len(ref_tokens) if len(ref_tokens) > 0 else 0
                    idf = math.log(num_docs / (df[ngram] + 1))
                    ref_tfidf[ngram] = tf * idf
                
                # Cosine similarity
                common_ngrams = set(hyp_tfidf.keys()) & set(ref_tfidf.keys())
                if not common_ngrams:
                    ref_scores.append(0.0)
                    continue
                
                dot_product = sum(hyp_tfidf[ng] * ref_tfidf[ng] for ng in common_ngrams)
                hyp_norm = math.sqrt(sum(v**2 for v in hyp_tfidf.values()))
                ref_norm = math.sqrt(sum(v**2 for v in ref_tfidf.values()))
                
                if hyp_norm > 0 and ref_norm > 0:
                    ref_scores.append(dot_product / (hyp_norm * ref_norm))
                else:
                    ref_scores.append(0.0)
            
            scores.append(np.mean(ref_scores) if ref_scores else 0.0)
        
        return np.mean(scores)
    
    def compute_rouge_l(self, references, hypotheses):
        """
        Compute ROUGE-L score (longest common subsequence)
        """
        def lcs_length(x, y):
            """Compute length of longest common subsequence"""
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        scores = []
        for ref_list, hyp in zip(references, hypotheses):
            hyp_tokens = hyp.split()
            ref_scores = []
            
            for ref in ref_list:
                ref_tokens = ref.split()
                lcs_len = lcs_length(ref_tokens, hyp_tokens)
                
                if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
                    ref_scores.append(0.0)
                    continue
                
                precision = lcs_len / len(hyp_tokens)
                recall = lcs_len / len(ref_tokens)
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    ref_scores.append(f1)
                else:
                    ref_scores.append(0.0)
            
            scores.append(max(ref_scores) if ref_scores else 0.0)
        
        return np.mean(scores)
    
    def evaluate(self, references, hypotheses):
        """
        Compute all metrics
        Args:
            references: List of lists of reference captions (strings)
            hypotheses: List of generated captions (strings)
        Returns:
            Dictionary of metric scores
        """
        # Tokenize for BLEU
        references_tokens = [
            [ref.split() for ref in ref_list]
            for ref_list in references
        ]
        hypotheses_tokens = [hyp.split() for hyp in hypotheses]
        
        # Compute metrics
        bleu_scores = self.compute_bleu(references_tokens, hypotheses_tokens)
        meteor = self.compute_meteor(references, hypotheses)
        cider = self.compute_cider(references, hypotheses)
        rouge_l = self.compute_rouge_l(references, hypotheses)
        
        results = {
            **bleu_scores,
            'METEOR': meteor,
            'CIDEr': cider,
            'ROUGE-L': rouge_l
        }
        
        return results


def evaluate_model(model, data_loader, vocab, device='cpu', max_samples=None):
    """
    Evaluate a model on a dataset
    Args:
        model: Captioning model
        data_loader: DataLoader for evaluation
        vocab: Vocabulary
        device: Device to run on
        max_samples: Maximum number of samples to evaluate (None for all)
    Returns:
        Dictionary with generated captions and references
    """
    model.eval()
    model = model.to(device)
    
    all_references = []
    all_hypotheses = []
    all_image_paths = []
    
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            images = batch['images']
            captions = batch['captions']
            image_paths = batch['image_paths']
            
            # Generate captions for each image
            for img, ref_caption, img_path in zip(images, captions, image_paths):
                # Preprocess image
                img_tensor = model.feature_extractor.preprocess(img).to(device)
                
                # Generate caption
                generated_caption, _ = model.generate_caption(img_tensor, vocab)
                
                all_hypotheses.append(generated_caption)
                all_references.append([ref_caption])  # Single reference per image
                all_image_paths.append(img_path)
                
                num_samples += 1
                if max_samples and num_samples >= max_samples:
                    break
            
            if max_samples and num_samples >= max_samples:
                break
    
    return {
        'hypotheses': all_hypotheses,
        'references': all_references,
        'image_paths': all_image_paths
    }


def compare_models(rnn_results, transformer_results, output_file=None):
    """
    Compare results from RNN and Transformer models
    """
    evaluator = CaptionEvaluator()
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    # Evaluate RNN
    print("\nRNN Model Results:")
    print("-" * 60)
    rnn_metrics = evaluator.evaluate(rnn_results['references'], rnn_results['hypotheses'])
    for metric, score in rnn_metrics.items():
        print(f"{metric:12s}: {score:.4f}")
    
    # Evaluate Transformer
    print("\nTransformer Model Results:")
    print("-" * 60)
    transformer_metrics = evaluator.evaluate(transformer_results['references'], transformer_results['hypotheses'])
    for metric, score in transformer_metrics.items():
        print(f"{metric:12s}: {score:.4f}")
    
    # Comparison
    print("\nImprovement (Transformer vs RNN):")
    print("-" * 60)
    for metric in rnn_metrics.keys():
        diff = transformer_metrics[metric] - rnn_metrics[metric]
        pct = (diff / rnn_metrics[metric] * 100) if rnn_metrics[metric] > 0 else 0
        print(f"{metric:12s}: {diff:+.4f} ({pct:+.2f}%)")
    
    # Sample comparisons
    print("\nSample Caption Comparisons:")
    print("-" * 60)
    for i in range(min(5, len(rnn_results['hypotheses']))):
        print(f"\nImage: {rnn_results['image_paths'][i]}")
        print(f"Reference:   {rnn_results['references'][i][0]}")
        print(f"RNN:         {rnn_results['hypotheses'][i]}")
        print(f"Transformer: {transformer_results['hypotheses'][i]}")
    
    # Save results
    if output_file:
        comparison = {
            'rnn_metrics': rnn_metrics,
            'transformer_metrics': transformer_metrics,
            'rnn_results': rnn_results,
            'transformer_results': transformer_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
    
    return rnn_metrics, transformer_metrics


if __name__ == "__main__":
    print("Evaluation metrics module loaded successfully!")

