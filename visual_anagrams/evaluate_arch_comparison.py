# evaluate_arch_comparison.py
"""
Architectural Comparison Evaluation for Visual Anagrams.

This script evaluates visual anagrams using both CNN-based (ResNet) and 
ViT-based (Vision Transformer) vision encoders to understand how different 
architectural inductive biases affect illusion perception.

Key Hypothesis:
- CNNs have spatial locality bias (local receptive fields)
- ViTs use global self-attention (all patches equally accessible)
- Visual anagram transformations may affect these architectures differently

Metrics Computed:
- Alignment Score (A) for both CNN and ViT
- Concealment Score (C) for both CNN and ViT  
- Architecture Gap (Δ_A, Δ_C): Difference in performance
- Cross-Architecture Agreement: How similarly they perceive the illusion
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms.functional as TF
import clip
from PIL import Image
from tqdm import tqdm
import json
import pandas as pd

# Import views to ensure classes are available when unpickling
try:
    from visual_anagrams.views import get_views
except ImportError:
    # May not be needed if running as part of package
    pass


def load_clip_models(device='cuda'):
    """
    Load both CNN-based (RN50) and ViT-based (ViT-B/32) CLIP models.
    
    Returns:
        dict with 'cnn' and 'vit' keys containing (model, preprocess) tuples
    """
    print("Loading CLIP models...")
    print("  - RN50 (CNN-based): ResNet-50 architecture")
    print("  - ViT-B/32 (Transformer-based): Vision Transformer")
    
    models = {}
    
    # Load CNN-based model (ResNet-50)
    cnn_model, cnn_preprocess = clip.load("RN50", device=device)
    models['cnn'] = (cnn_model, cnn_preprocess)
    print("  ✓ Loaded RN50")
    
    # Load ViT-based model (currently used in standard evaluation)
    vit_model, vit_preprocess = clip.load("ViT-B/32", device=device)
    models['vit'] = (vit_model, vit_preprocess)
    print("  ✓ Loaded ViT-B/32")
    
    return models


def compute_score_matrix_with_arch(illusion_path, views, prompts, clip_model, clip_preprocess, device='cuda'):
    """
    Compute CLIP score matrix for a specific architecture.
    
    Same as standard evaluation but parameterized by model.
    """
    N = len(views)
    assert N == len(prompts), "Number of views must match number of prompts"
    
    # Load the illusion image as PIL Image
    illusion_pil = Image.open(illusion_path).convert('RGB')
    
    # Convert to tensor in range [-1, 1] (as used in training)
    illusion_tensor = TF.to_tensor(illusion_pil) * 2 - 1
    illusion_tensor = illusion_tensor.to(device)
    
    # Get image embeddings for each view
    image_embeddings = []
    
    for view in views:
        try:
            # Apply view transformation to the tensor
            viewed_tensor = view.view(illusion_tensor)
            
            # Convert back to PIL Image for CLIP
            viewed_normalized = (viewed_tensor + 1) / 2
            viewed_normalized = torch.clamp(viewed_normalized, 0, 1)
            
            viewed_pil = TF.to_pil_image(viewed_normalized.cpu())
            
            # Preprocess for CLIP
            image_input = clip_preprocess(viewed_pil).unsqueeze(0).to(device)
            
            # Get CLIP embedding
            with torch.no_grad():
                image_features = clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            image_embeddings.append(image_features)
            
        except Exception as e:
            print(f"Error processing view {view.__class__.__name__}: {e}")
            raise
    
    # Stack image embeddings [N, embed_dim]
    image_embeddings = torch.cat(image_embeddings, dim=0)
    
    # Get text embeddings
    text_tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Compute score matrix S_ij = φ_img(v_i(x))^T φ_text(p_j)
    score_matrix = (image_embeddings @ text_features.T).cpu().numpy()
    
    return score_matrix


def compute_alignment_score(score_matrix):
    """Compute alignment score A = min diag(S)"""
    diagonal = np.diag(score_matrix)
    return float(np.min(diagonal))


def compute_concealment_score(score_matrix, temperature=0.01):
    """Compute concealment score C = (1/N) * tr(softmax(S/τ))"""
    N = score_matrix.shape[0]
    
    score_tensor = torch.from_numpy(score_matrix).float()
    
    # Compute softmax over rows
    row_softmax = torch.softmax(score_tensor / temperature, dim=1)
    row_score = torch.trace(row_softmax) / N
    
    # Compute softmax over columns
    col_softmax = torch.softmax(score_tensor / temperature, dim=0)
    col_score = torch.trace(col_softmax) / N
    
    # Average both directions
    concealment_score = (row_score + col_score) / 2.0
    
    return float(concealment_score)


def compute_cross_architecture_agreement(cnn_matrix, vit_matrix):
    """
    Compute how similarly CNN and ViT perceive the illusion.
    
    Returns:
        - agreement: Fraction of views where top prediction matches
        - correlation: Correlation between score matrices
    """
    N = cnn_matrix.shape[0]
    
    # Check if top predictions match for each view
    cnn_top = np.argmax(cnn_matrix, axis=1)
    vit_top = np.argmax(vit_matrix, axis=1)
    agreement = np.mean(cnn_top == vit_top)
    
    # Compute correlation between flattened matrices
    correlation = np.corrcoef(cnn_matrix.flatten(), vit_matrix.flatten())[0, 1]
    
    return {
        'agreement': float(agreement),
        'correlation': float(correlation),
        'cnn_top_predictions': cnn_top.tolist(),
        'vit_top_predictions': vit_top.tolist()
    }


def evaluate_sample_with_architectures(illusion_path, views, prompts, models, device='cuda'):
    """
    Evaluate a single illusion sample with both CNN and ViT architectures.
    
    Returns:
        dict with results for both architectures and comparison metrics
    """
    results = {}
    
    # Evaluate with CNN (ResNet-50)
    cnn_model, cnn_preprocess = models['cnn']
    cnn_matrix = compute_score_matrix_with_arch(
        illusion_path, views, prompts, cnn_model, cnn_preprocess, device
    )
    cnn_alignment = compute_alignment_score(cnn_matrix)
    cnn_concealment = compute_concealment_score(cnn_matrix)
    
    results['cnn'] = {
        'alignment_score': cnn_alignment,
        'concealment_score': cnn_concealment,
        'score_matrix': cnn_matrix.tolist(),
        'diagonal_scores': np.diag(cnn_matrix).tolist()
    }
    
    # Evaluate with ViT
    vit_model, vit_preprocess = models['vit']
    vit_matrix = compute_score_matrix_with_arch(
        illusion_path, views, prompts, vit_model, vit_preprocess, device
    )
    vit_alignment = compute_alignment_score(vit_matrix)
    vit_concealment = compute_concealment_score(vit_matrix)
    
    results['vit'] = {
        'alignment_score': vit_alignment,
        'concealment_score': vit_concealment,
        'score_matrix': vit_matrix.tolist(),
        'diagonal_scores': np.diag(vit_matrix).tolist()
    }
    
    # Compute architecture comparison metrics
    results['architecture_gap'] = {
        'delta_alignment': vit_alignment - cnn_alignment,
        'delta_concealment': vit_concealment - cnn_concealment,
    }
    
    # Compute cross-architecture agreement
    results['cross_architecture'] = compute_cross_architecture_agreement(cnn_matrix, vit_matrix)
    
    return results


def evaluate_results_directory(results_dir, image_size=1024, device='cuda', verbose=True):
    """
    Evaluate all samples in a results directory with architecture comparison.
    
    Args:
        results_dir: Path to results directory (e.g., results/flip.dog_cat.variant3)
        image_size: Size of images to evaluate (64, 256, or 1024)
        device: torch device
        verbose: Whether to print progress information
    
    Returns:
        dict with statistics and per-sample results
    """
    results_dir = Path(results_dir)
    
    # Load metadata (single file at root)
    metadata_path = results_dir / 'metadata.pkl'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    views = metadata['views']
    args = metadata['args']
    
    # Reconstruct prompts
    prompts = [f'{args.style} {p}'.strip() for p in args.prompts]
    
    if verbose:
        print(f"Evaluating results in: {results_dir}")
        print(f"Views: {[v.__class__.__name__ for v in views]}")
        print(f"Prompts: {prompts}")
        print(f"Image size: {image_size}x{image_size}")
    
    # Load models
    models = load_clip_models(device)
    
    # Find all sample directories (numbered)
    sample_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    
    if len(sample_dirs) == 0:
        raise ValueError(f"No sample directories found in {results_dir}")
    
    if verbose:
        print(f"Found {len(sample_dirs)} samples to evaluate")
    
    # Evaluate each sample
    per_sample_results = []
    failed_samples = []
    
    for sample_dir in tqdm(sample_dirs, desc="Evaluating samples", disable=not verbose):
        image_path = sample_dir / f'sample_{image_size}.png'
        
        if not image_path.exists():
            if verbose:
                print(f"\nWarning: Image not found at {image_path}, skipping...")
            failed_samples.append(str(sample_dir))
            continue
        
        try:
            sample_results = evaluate_sample_with_architectures(
                image_path, views, prompts, models, device
            )
            
            sample_results['sample_dir'] = sample_dir.name
            sample_results['image_path'] = str(image_path)
            per_sample_results.append(sample_results)
            
        except Exception as e:
            if verbose:
                print(f"\nError evaluating {image_path}:")
                print(f"  {type(e).__name__}: {e}")
            failed_samples.append(str(sample_dir))
            import traceback
            if verbose:
                traceback.print_exc()
            continue
    
    if len(per_sample_results) == 0:
        if failed_samples:
            raise ValueError(f"No samples were successfully evaluated. Failed samples: {failed_samples}")
        else:
            raise ValueError("No samples were successfully evaluated")
    
    if failed_samples and verbose:
        print(f"\nWarning: {len(failed_samples)} samples failed to evaluate: {failed_samples}")
    
    # Compute statistics
    # Statistics for CNN
    cnn_alignment_scores = [r['cnn']['alignment_score'] for r in per_sample_results]
    cnn_concealment_scores = [r['cnn']['concealment_score'] for r in per_sample_results]
    
    # Statistics for ViT
    vit_alignment_scores = [r['vit']['alignment_score'] for r in per_sample_results]
    vit_concealment_scores = [r['vit']['concealment_score'] for r in per_sample_results]
    
    # Architecture gap statistics
    delta_alignment_scores = [r['architecture_gap']['delta_alignment'] for r in per_sample_results]
    delta_concealment_scores = [r['architecture_gap']['delta_concealment'] for r in per_sample_results]
    
    # Cross-architecture agreement
    agreement_scores = [r['cross_architecture']['agreement'] for r in per_sample_results]
    correlation_scores = [r['cross_architecture']['correlation'] for r in per_sample_results]
    
    statistics = {
        'num_samples': len(per_sample_results),
        'num_failed': len(failed_samples),
        'cnn': {
            'alignment': compute_stats(cnn_alignment_scores),
            'concealment': compute_stats(cnn_concealment_scores),
        },
        'vit': {
            'alignment': compute_stats(vit_alignment_scores),
            'concealment': compute_stats(vit_concealment_scores),
        },
        'architecture_gap': {
            'delta_alignment': compute_stats(delta_alignment_scores),
            'delta_concealment': compute_stats(delta_concealment_scores),
        },
        'cross_architecture': {
            'agreement': compute_stats(agreement_scores),
            'correlation': compute_stats(correlation_scores),
        }
    }
    
    evaluation_results = {
        'metadata': {
            'results_dir': str(results_dir),
            'image_size': image_size,
            'views': [v.__class__.__name__ for v in views],
            'prompts': prompts,
            'evaluation_type': 'architecture_comparison',
            'architectures': ['RN50 (CNN)', 'ViT-B/32 (Transformer)']
        },
        'statistics': statistics,
        'per_sample_results': per_sample_results,
        'failed_samples': failed_samples
    }
    
    return evaluation_results


def compute_stats(values):
    """Compute statistics for a list of values."""
    values = np.array(values)
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
        'q90': float(np.percentile(values, 90)),
        'q95': float(np.percentile(values, 95)),
    }


def print_statistics(evaluation_results):
    """Print evaluation statistics in a formatted table."""
    stats = evaluation_results['statistics']
    metadata = evaluation_results['metadata']
    
    print("\n" + "="*100)
    print("ARCHITECTURE COMPARISON EVALUATION RESULTS")
    print("="*100)
    print(f"\nResults Directory: {metadata['results_dir']}")
    print(f"Views: {metadata['views']}")
    print(f"Prompts: {metadata['prompts']}")
    print(f"Image Size: {metadata['image_size']}x{metadata['image_size']}")
    print(f"Number of Samples: {stats['num_samples']}")
    if stats['num_failed'] > 0:
        print(f"Number of Failed: {stats['num_failed']}")
    
    print("\n" + "="*100)
    print("CNN (ResNet-50) PERFORMANCE")
    print("="*100)
    print(f"{'Metric':>30} | {'Alignment (A)':>15} | {'Concealment (C)':>15}")
    print("-"*100)
    print(f"{'Mean':>30} | {stats['cnn']['alignment']['mean']:>15.4f} | {stats['cnn']['concealment']['mean']:>15.4f}")
    print(f"{'Std Dev':>30} | {stats['cnn']['alignment']['std']:>15.4f} | {stats['cnn']['concealment']['std']:>15.4f}")
    print(f"{'90th Percentile':>30} | {stats['cnn']['alignment']['q90']:>15.4f} | {stats['cnn']['concealment']['q90']:>15.4f}")
    print(f"{'95th Percentile':>30} | {stats['cnn']['alignment']['q95']:>15.4f} | {stats['cnn']['concealment']['q95']:>15.4f}")
    
    print("\n" + "="*100)
    print("ViT (Vision Transformer) PERFORMANCE")
    print("="*100)
    print(f"{'Metric':>30} | {'Alignment (A)':>15} | {'Concealment (C)':>15}")
    print("-"*100)
    print(f"{'Mean':>30} | {stats['vit']['alignment']['mean']:>15.4f} | {stats['vit']['concealment']['mean']:>15.4f}")
    print(f"{'Std Dev':>30} | {stats['vit']['alignment']['std']:>15.4f} | {stats['vit']['concealment']['std']:>15.4f}")
    print(f"{'90th Percentile':>30} | {stats['vit']['alignment']['q90']:>15.4f} | {stats['vit']['concealment']['q90']:>15.4f}")
    print(f"{'95th Percentile':>30} | {stats['vit']['alignment']['q95']:>15.4f} | {stats['vit']['concealment']['q95']:>15.4f}")
    
    print("\n" + "="*100)
    print("ARCHITECTURE GAP (ViT - CNN)")
    print("="*100)
    print(f"{'Metric':>30} | {'Δ Alignment':>15} | {'Δ Concealment':>15}")
    print("-"*100)
    print(f"{'Mean':>30} | {stats['architecture_gap']['delta_alignment']['mean']:>15.4f} | {stats['architecture_gap']['delta_concealment']['mean']:>15.4f}")
    print(f"{'Std Dev':>30} | {stats['architecture_gap']['delta_alignment']['std']:>15.4f} | {stats['architecture_gap']['delta_concealment']['std']:>15.4f}")
    
    print("\n" + "="*100)
    print("CROSS-ARCHITECTURE AGREEMENT")
    print("="*100)
    print(f"Mean Agreement (top-1 match): {stats['cross_architecture']['agreement']['mean']:.4f}")
    print(f"Mean Correlation (score matrices): {stats['cross_architecture']['correlation']['mean']:.4f}")
    
    # Interpretation
    print("\n" + "="*100)
    print("INTERPRETATION")
    print("="*100)
    
    delta_a = stats['architecture_gap']['delta_alignment']['mean']
    delta_c = stats['architecture_gap']['delta_concealment']['mean']
    
    if delta_a > 0.01:
        print(f"✓ ViT maintains better alignment (Δ_A = +{delta_a:.4f})")
        print("  → Suggests global attention better preserves illusion structure")
    elif delta_a < -0.01:
        print(f"✗ CNN maintains better alignment (Δ_A = {delta_a:.4f})")
        print("  → Suggests local features dominate in this illusion")
    else:
        print(f"≈ Similar alignment performance (Δ_A = {delta_a:.4f})")
    
    if delta_c > 0.01:
        print(f"✓ ViT has better concealment (Δ_C = +{delta_c:.4f})")
        print("  → CNN leaks more information via local artifacts")
    elif delta_c < -0.01:
        print(f"✗ CNN has better concealment (Δ_C = {delta_c:.4f})")
        print("  → ViT may leak via global structure")
    else:
        print(f"≈ Similar concealment performance (Δ_C = {delta_c:.4f})")
    
    print("="*100 + "\n")


def save_results(evaluation_results, output_path):
    """Save evaluation results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def save_results_csv(evaluation_results, output_path):
    """Save per-sample results to CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame from per-sample results
    data = []
    for result in evaluation_results['per_sample_results']:
        row = {
            'sample_dir': result['sample_dir'],
            'cnn_alignment': result['cnn']['alignment_score'],
            'cnn_concealment': result['cnn']['concealment_score'],
            'vit_alignment': result['vit']['alignment_score'],
            'vit_concealment': result['vit']['concealment_score'],
            'delta_alignment': result['architecture_gap']['delta_alignment'],
            'delta_concealment': result['architecture_gap']['delta_concealment'],
            'agreement': result['cross_architecture']['agreement'],
            'correlation': result['cross_architecture']['correlation'],
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    print(f"Per-sample results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Visual Anagrams with CNN vs ViT architecture comparison"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to results directory (e.g., results/flip.dog_cat.variant3)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=1024,
        choices=[64, 256, 1024],
        help="Size of images to evaluate"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save JSON results (optional)"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to save CSV results (optional)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--quiet",
        action='store_true',
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluation_results = evaluate_results_directory(
        args.results_dir,
        image_size=args.image_size,
        device=args.device,
        verbose=not args.quiet
    )
    
    # Print statistics
    if not args.quiet:
        print_statistics(evaluation_results)
    
    # Save results if requested
    if args.output_json:
        save_results(evaluation_results, args.output_json)
    
    if args.output_csv:
        save_results_csv(evaluation_results, args.output_csv)
    
    # Also save to default location in results directory
    if not args.quiet:
        default_json_path = Path(args.results_dir) / f'evaluation_arch_{args.image_size}.json'
        save_results(evaluation_results, default_json_path)
        
        default_csv_path = Path(args.results_dir) / f'evaluation_arch_{args.image_size}.csv'
        save_results_csv(evaluation_results, default_csv_path)


if __name__ == '__main__':
    main()