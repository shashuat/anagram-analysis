# evaluate.py
"""
Evaluation script for Visual Anagrams illusions using CLIP metrics.

Implements metrics from Section 4.1 of the paper:
- Alignment Score (A): min diag(S) - worst alignment across all views
- Concealment Score (C): (1/N) * tr(softmax(S/τ)) - how well views are concealed
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


def load_clip_model(device='cuda'):
    """Load CLIP model for evaluation."""
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess


def compute_score_matrix(illusion_path, views, prompts, clip_model, clip_preprocess, device='cuda'):
    """
    Compute CLIP score matrix S where S_ij = φ_img(v_i(x))^T φ_text(p_j)
    
    Args:
        illusion_path: Path to the illusion image
        views: List of view transformation objects
        prompts: List of text prompts
        clip_model: CLIP model
        clip_preprocess: CLIP preprocessing function
        device: torch device
    
    Returns:
        score_matrix: N×N matrix of CLIP similarities
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
            # First normalize from [-1, 1] to [0, 1]
            viewed_normalized = (viewed_tensor + 1) / 2
            viewed_normalized = torch.clamp(viewed_normalized, 0, 1)
            
            # Convert to PIL
            viewed_pil = TF.to_pil_image(viewed_normalized.cpu())
            
            # Preprocess for CLIP
            image_input = clip_preprocess(viewed_pil).unsqueeze(0).to(device)
            
            # Get CLIP embedding
            with torch.no_grad():
                image_features = clip_model.encode_image(image_input)
                # Normalize to unit norm
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
        # Normalize to unit norm
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Compute score matrix S_ij = φ_img(v_i(x))^T φ_text(p_j)
    score_matrix = (image_embeddings @ text_features.T).cpu().numpy()
    
    return score_matrix


def compute_alignment_score(score_matrix):
    """
    Compute alignment score A = min diag(S)
    Measures the worst alignment of all views.
    """
    diagonal = np.diag(score_matrix)
    return float(np.min(diagonal))


def compute_concealment_score(score_matrix, temperature=0.01):
    """
    Compute concealment score C = (1/N) * tr(softmax(S/τ))
    Measures how well CLIP can classify a view as one of the N prompts.
    
    We average both directions of the softmax as mentioned in the paper.
    """
    N = score_matrix.shape[0]
    
    # Convert to tensor and ensure float32 (trace doesn't work with float16)
    score_tensor = torch.from_numpy(score_matrix).float()
    
    # Compute softmax over rows (for each view, prob over prompts)
    row_softmax = torch.softmax(score_tensor / temperature, dim=1)
    row_score = torch.trace(row_softmax) / N
    
    # Compute softmax over columns (for each prompt, prob over views)
    col_softmax = torch.softmax(score_tensor / temperature, dim=0)
    col_score = torch.trace(col_softmax) / N
    
    # Average both directions
    concealment_score = (row_score + col_score) / 2.0
    
    return float(concealment_score)


def evaluate_sample(illusion_path, views, prompts, clip_model, clip_preprocess, device='cuda'):
    """
    Evaluate a single illusion sample.
    
    Returns:
        dict with 'alignment_score', 'concealment_score', and 'score_matrix'
    """
    score_matrix = compute_score_matrix(
        illusion_path, views, prompts, clip_model, clip_preprocess, device
    )
    
    alignment_score = compute_alignment_score(score_matrix)
    concealment_score = compute_concealment_score(score_matrix)
    
    return {
        'alignment_score': alignment_score,
        'concealment_score': concealment_score,
        'score_matrix': score_matrix.tolist(),
        'diagonal_scores': np.diag(score_matrix).tolist()
    }


def evaluate_results_directory(results_dir, image_size=1024, device='cuda', verbose=True):
    """
    Evaluate all samples in a results directory.
    
    Args:
        results_dir: Path to results directory (e.g., results/flip.dog_cat.variant3)
        image_size: Size of images to evaluate (64, 256, or 1024)
        device: torch device
        verbose: Whether to print progress information
    
    Returns:
        dict with statistics and per-sample results
    """
    results_dir = Path(results_dir)
    
    # Load metadata
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
    
    # Load CLIP model
    if verbose:
        print("Loading CLIP model...")
    clip_model, clip_preprocess = load_clip_model(device)
    
    # Find all sample directories
    sample_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    
    if len(sample_dirs) == 0:
        raise ValueError(f"No sample directories found in {results_dir}")
    
    if verbose:
        print(f"Found {len(sample_dirs)} samples to evaluate")
    
    # Evaluate each sample
    results = []
    alignment_scores = []
    concealment_scores = []
    failed_samples = []
    
    for sample_dir in tqdm(sample_dirs, desc="Evaluating samples", disable=not verbose):
        image_path = sample_dir / f'sample_{image_size}.png'
        
        if not image_path.exists():
            if verbose:
                print(f"Warning: Image not found at {image_path}, skipping...")
            failed_samples.append(str(sample_dir))
            continue
        
        try:
            sample_result = evaluate_sample(
                image_path, views, prompts, clip_model, clip_preprocess, device
            )
            
            sample_result['sample_dir'] = sample_dir.name
            sample_result['image_path'] = str(image_path)
            results.append(sample_result)
            
            alignment_scores.append(sample_result['alignment_score'])
            concealment_scores.append(sample_result['concealment_score'])
            
        except Exception as e:
            if verbose:
                print(f"\nError evaluating {image_path}:")
                print(f"  {type(e).__name__}: {e}")
            failed_samples.append(str(sample_dir))
            import traceback
            if verbose:
                traceback.print_exc()
            continue
    
    if len(results) == 0:
        if failed_samples:
            raise ValueError(f"No samples were successfully evaluated. Failed samples: {failed_samples}")
        else:
            raise ValueError("No samples were successfully evaluated")
    
    if failed_samples and verbose:
        print(f"\nWarning: {len(failed_samples)} samples failed to evaluate: {failed_samples}")
    
    # Compute statistics
    alignment_scores = np.array(alignment_scores)
    concealment_scores = np.array(concealment_scores)
    
    statistics = {
        'num_samples': len(results),
        'num_failed': len(failed_samples),
        'alignment': {
            'mean': float(np.mean(alignment_scores)),
            'std': float(np.std(alignment_scores)),
            'min': float(np.min(alignment_scores)),
            'max': float(np.max(alignment_scores)),
            'median': float(np.median(alignment_scores)),
            'q90': float(np.percentile(alignment_scores, 90)),
            'q95': float(np.percentile(alignment_scores, 95)),
        },
        'concealment': {
            'mean': float(np.mean(concealment_scores)),
            'std': float(np.std(concealment_scores)),
            'min': float(np.min(concealment_scores)),
            'max': float(np.max(concealment_scores)),
            'median': float(np.median(concealment_scores)),
            'q90': float(np.percentile(concealment_scores, 90)),
            'q95': float(np.percentile(concealment_scores, 95)),
        }
    }
    
    return {
        'statistics': statistics,
        'per_sample_results': results,
        'failed_samples': failed_samples,
        'metadata': {
            'results_dir': str(results_dir),
            'views': [v.__class__.__name__ for v in views],
            'prompts': prompts,
            'image_size': image_size,
        }
    }


def print_statistics(evaluation_results):
    """Pretty print evaluation statistics."""
    stats = evaluation_results['statistics']
    metadata = evaluation_results['metadata']
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nResults Directory: {metadata['results_dir']}")
    print(f"Views: {metadata['views']}")
    print(f"Prompts: {metadata['prompts']}")
    print(f"Image Size: {metadata['image_size']}x{metadata['image_size']}")
    print(f"Number of Samples: {stats['num_samples']}")
    if stats['num_failed'] > 0:
        print(f"Number of Failed: {stats['num_failed']}")
    
    print(f"\n{'Metric':>30} | {'Alignment (A)':>15} | {'Concealment (C)':>15}")
    print("-"*80)
    print(f"{'Mean':>30} | {stats['alignment']['mean']:>15.4f} | {stats['concealment']['mean']:>15.4f}")
    print(f"{'Std Dev':>30} | {stats['alignment']['std']:>15.4f} | {stats['concealment']['std']:>15.4f}")
    print(f"{'Min':>30} | {stats['alignment']['min']:>15.4f} | {stats['concealment']['min']:>15.4f}")
    print(f"{'Max':>30} | {stats['alignment']['max']:>15.4f} | {stats['concealment']['max']:>15.4f}")
    print(f"{'Median':>30} | {stats['alignment']['median']:>15.4f} | {stats['concealment']['median']:>15.4f}")
    print(f"{'90th Percentile':>30} | {stats['alignment']['q90']:>15.4f} | {stats['concealment']['q90']:>15.4f}")
    print(f"{'95th Percentile':>30} | {stats['alignment']['q95']:>15.4f} | {stats['concealment']['q95']:>15.4f}")
    print("="*80 + "\n")
    
    # Print best samples
    if len(evaluation_results['per_sample_results']) > 0:
        print("Top 3 samples by alignment score:")
        sorted_results = sorted(
            evaluation_results['per_sample_results'],
            key=lambda x: x['alignment_score'],
            reverse=True
        )
        for i, result in enumerate(sorted_results[:3], 1):
            print(f"  {i}. {result['sample_dir']}: A={result['alignment_score']:.4f}, C={result['concealment_score']:.4f}")
        print()


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
            'alignment_score': result['alignment_score'],
            'concealment_score': result['concealment_score'],
        }
        # Add individual view scores
        for i, score in enumerate(result['diagonal_scores']):
            row[f'view_{i}_score'] = score
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    print(f"Per-sample results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Visual Anagrams illusions using CLIP metrics"
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
        default_json_path = Path(args.results_dir) / f'evaluation_{args.image_size}.json'
        save_results(evaluation_results, default_json_path)
        
        default_csv_path = Path(args.results_dir) / f'evaluation_{args.image_size}.csv'
        save_results_csv(evaluation_results, default_csv_path)


if __name__ == '__main__':
    main()