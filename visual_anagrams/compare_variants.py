# compare_variants.py
"""
Compare evaluation results across multiple variants.
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_evaluation_results(results_dirs, image_size=1024):
    """
    Load evaluation results from multiple directories.
    
    Args:
        results_dirs: List of result directory paths
        image_size: Image size to load results for
    
    Returns:
        dict mapping variant name to evaluation results
    """
    all_results = {}
    
    for results_dir in results_dirs:
        results_dir = Path(results_dir)
        eval_path = results_dir / f'evaluation_{image_size}.json'
        
        if not eval_path.exists():
            print(f"Warning: Evaluation results not found at {eval_path}")
            print(f"Run evaluate.py first on {results_dir}")
            continue
        
        with open(eval_path, 'r') as f:
            results = json.load(f)
        
        variant_name = results_dir.name
        all_results[variant_name] = results
    
    return all_results


def create_comparison_table(all_results):
    """Create a comparison table of statistics across variants."""
    data = []
    
    for variant_name, results in all_results.items():
        stats = results['statistics']
        data.append({
            'Variant': variant_name,
            'N': stats['num_samples'],
            'A (mean)': stats['alignment']['mean'],
            'A_0.9': stats['alignment']['q90'],
            'A_0.95': stats['alignment']['q95'],
            'C (mean)': stats['concealment']['mean'],
            'C_0.9': stats['concealment']['q90'],
            'C_0.95': stats['concealment']['q95'],
        })
    
    df = pd.DataFrame(data)
    return df


def plot_score_distributions(all_results, output_dir=None):
    """Plot distribution of scores across variants."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Alignment scores
    ax = axes[0]
    for variant_name, results in all_results.items():
        scores = [r['alignment_score'] for r in results['per_sample_results']]
        ax.hist(scores, alpha=0.6, label=variant_name, bins=20)
    ax.set_xlabel('Alignment Score (A)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Alignment Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Concealment scores
    ax = axes[1]
    for variant_name, results in all_results.items():
        scores = [r['concealment_score'] for r in results['per_sample_results']]
        ax.hist(scores, alpha=0.6, label=variant_name, bins=20)
    ax.set_xlabel('Concealment Score (C)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Concealment Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'score_distributions.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()


def plot_scatter_comparison(all_results, output_dir=None):
    """Plot alignment vs concealment scores for each variant."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for variant_name, results in all_results.items():
        alignment_scores = [r['alignment_score'] for r in results['per_sample_results']]
        concealment_scores = [r['concealment_score'] for r in results['per_sample_results']]
        
        ax.scatter(alignment_scores, concealment_scores, 
                  alpha=0.6, label=variant_name, s=50)
    
    ax.set_xlabel('Alignment Score (A)', fontsize=12)
    ax.set_ylabel('Concealment Score (C)', fontsize=12)
    ax.set_title('Alignment vs Concealment Scores', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add diagonal line for reference
    lims = [
        max(ax.get_xlim()[0], ax.get_ylim()[0]),
        min(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'scatter_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compare evaluation results across multiple variants"
    )
    parser.add_argument(
        "--results_dirs",
        type=str,
        nargs='+',
        required=True,
        help="Paths to result directories to compare"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=1024,
        choices=[64, 256, 1024],
        help="Size of images that were evaluated"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save comparison plots and tables"
    )
    
    args = parser.parse_args()
    
    # Load results
    print("Loading evaluation results...")
    all_results = load_evaluation_results(args.results_dirs, args.image_size)
    
    if len(all_results) == 0:
        print("No evaluation results found. Run evaluate.py first.")
        return
    
    print(f"\nLoaded results for {len(all_results)} variants")
    
    # Create comparison table
    print("\n" + "="*100)
    print("COMPARISON TABLE")
    print("="*100)
    comparison_df = create_comparison_table(all_results)
    print(comparison_df.to_string(index=False))
    print("="*100 + "\n")
    
    # Save table if output directory provided
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = output_dir / 'comparison_table.csv'
        comparison_df.to_csv(csv_path, index=False)
        print(f"Comparison table saved to: {csv_path}")
    
    # Create plots
    print("Generating comparison plots...")
    plot_score_distributions(all_results, args.output_dir)
    plot_scatter_comparison(all_results, args.output_dir)


if __name__ == '__main__':
    main()