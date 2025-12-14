# compare_arch_variants.py
"""
Compare architecture effects (CNN vs ViT) across multiple variants.

This script analyzes how different generation variants (variant0, variant1, etc.)
are affected differently by architectural biases in CNN vs ViT encoders.
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for batch processing
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import views to ensure classes are available when loading results
try:
    from visual_anagrams.views import get_views
except ImportError:
    # May not be needed
    pass


def load_arch_evaluation_results(results_dirs, image_size=1024):
    """
    Load architecture comparison evaluation results from multiple directories.
    
    Args:
        results_dirs: List of result directory paths
        image_size: Image size to load results for
    
    Returns:
        dict mapping variant name to evaluation results
    """
    all_results = {}
    
    for results_dir in results_dirs:
        results_dir = Path(results_dir)
        eval_path = results_dir / f'evaluation_arch_{image_size}.json'
        
        if not eval_path.exists():
            print(f"Warning: Architecture evaluation results not found at {eval_path}")
            print(f"Run evaluate_arch_comparison.py first on {results_dir}")
            continue
        
        with open(eval_path, 'r') as f:
            results = json.load(f)
        
        variant_name = results_dir.name
        all_results[variant_name] = results
    
    return all_results


def create_comparison_table(all_results):
    """Create a comparison table showing CNN vs ViT performance across variants."""
    data = []
    
    for variant_name, results in all_results.items():
        stats = results['statistics']
        
        row = {
            'Variant': variant_name,
            'N': stats['num_samples'],
            # CNN scores
            'CNN_A': stats['cnn']['alignment']['mean'],
            'CNN_A90': stats['cnn']['alignment']['q90'],
            'CNN_C': stats['cnn']['concealment']['mean'],
            'CNN_C90': stats['cnn']['concealment']['q90'],
            # ViT scores
            'ViT_A': stats['vit']['alignment']['mean'],
            'ViT_A90': stats['vit']['alignment']['q90'],
            'ViT_C': stats['vit']['concealment']['mean'],
            'ViT_C90': stats['vit']['concealment']['q90'],
            # Architecture gap
            'Δ_A': stats['architecture_gap']['delta_alignment']['mean'],
            'Δ_C': stats['architecture_gap']['delta_concealment']['mean'],
            # Agreement
            'Agreement': stats['cross_architecture']['agreement']['mean'],
            'Correlation': stats['cross_architecture']['correlation']['mean'],
        }
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def plot_architecture_gap_comparison(all_results, output_dir=None):
    """
    Plot architecture gap (ViT - CNN) for alignment and concealment across variants.
    
    This shows which variants benefit more from ViT vs CNN architecture.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    variants = []
    delta_alignment = []
    delta_concealment = []
    
    for variant_name, results in sorted(all_results.items()):
        stats = results['statistics']
        variants.append(variant_name)
        delta_alignment.append(stats['architecture_gap']['delta_alignment']['mean'])
        delta_concealment.append(stats['architecture_gap']['delta_concealment']['mean'])
    
    x = np.arange(len(variants))
    width = 0.35
    
    # Alignment gap
    ax = axes[0]
    bars = ax.bar(x, delta_alignment, width, 
                   color=['green' if d > 0 else 'red' for d in delta_alignment],
                   alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Variant', fontsize=12)
    ax.set_ylabel('Δ Alignment (ViT - CNN)', fontsize=12)
    ax.set_title('Architecture Gap: Alignment Score', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, delta_alignment)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom' if val > 0 else 'top', fontsize=9)
    
    # Concealment gap
    ax = axes[1]
    bars = ax.bar(x, delta_concealment, width,
                   color=['green' if d > 0 else 'red' for d in delta_concealment],
                   alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Variant', fontsize=12)
    ax.set_ylabel('Δ Concealment (ViT - CNN)', fontsize=12)
    ax.set_title('Architecture Gap: Concealment Score', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, delta_concealment)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom' if val > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'architecture_gap_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.close()  # Close instead of show for batch processing


def plot_cnn_vs_vit_scatter(all_results, output_dir=None):
    """
    Scatter plot showing CNN vs ViT performance for each variant.
    
    Points above diagonal = ViT better, below = CNN better.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Alignment comparison
    ax = axes[0]
    for variant_name, results in all_results.items():
        stats = results['statistics']
        cnn_a = stats['cnn']['alignment']['mean']
        vit_a = stats['vit']['alignment']['mean']
        
        ax.scatter(cnn_a, vit_a, s=100, alpha=0.7, label=variant_name)
    
    # Add diagonal line (perfect agreement)
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='CNN = ViT')
    
    ax.set_xlabel('CNN Alignment Score', fontsize=12)
    ax.set_ylabel('ViT Alignment Score', fontsize=12)
    ax.set_title('Alignment: CNN vs ViT', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.text(0.05, 0.95, 'Above diagonal → ViT better\nBelow diagonal → CNN better',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Concealment comparison
    ax = axes[1]
    for variant_name, results in all_results.items():
        stats = results['statistics']
        cnn_c = stats['cnn']['concealment']['mean']
        vit_c = stats['vit']['concealment']['mean']
        
        ax.scatter(cnn_c, vit_c, s=100, alpha=0.7, label=variant_name)
    
    # Add diagonal line
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='CNN = ViT')
    
    ax.set_xlabel('CNN Concealment Score', fontsize=12)
    ax.set_ylabel('ViT Concealment Score', fontsize=12)
    ax.set_title('Concealment: CNN vs ViT', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.text(0.05, 0.95, 'Above diagonal → ViT better\nBelow diagonal → CNN better',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'cnn_vs_vit_scatter.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.close()  # Close instead of show for batch processing


def plot_agreement_heatmap(all_results, output_dir=None):
    """
    Heatmap showing cross-architecture agreement and correlation across variants.
    """
    variants = sorted(all_results.keys())
    
    agreement_data = []
    correlation_data = []
    
    for variant_name in variants:
        stats = all_results[variant_name]['statistics']
        agreement_data.append(stats['cross_architecture']['agreement']['mean'])
        correlation_data.append(stats['cross_architecture']['correlation']['mean'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Agreement plot
    ax = axes[0]
    bars = ax.barh(variants, agreement_data, color='skyblue', alpha=0.7)
    ax.set_xlabel('Agreement (Top-1 Match Fraction)', fontsize=12)
    ax.set_ylabel('Variant', fontsize=12)
    ax.set_title('Cross-Architecture Agreement', fontsize=14)
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, val in zip(bars, agreement_data):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{val:.3f}',
                ha='left', va='center', fontsize=10)
    
    # Correlation plot
    ax = axes[1]
    bars = ax.barh(variants, correlation_data, color='lightcoral', alpha=0.7)
    ax.set_xlabel('Correlation (Score Matrices)', fontsize=12)
    ax.set_ylabel('Variant', fontsize=12)
    ax.set_title('Cross-Architecture Correlation', fontsize=14)
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, val in zip(bars, correlation_data):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{val:.3f}',
                ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'agreement_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.close()  # Close instead of show for batch processing


def plot_architecture_sensitivity(all_results, output_dir=None):
    """
    Plot showing which variants are most sensitive to architecture choice.
    
    Sensitivity = standard deviation of architecture gap across samples
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    variants = []
    alignment_sensitivity = []
    concealment_sensitivity = []
    
    for variant_name, results in sorted(all_results.items()):
        stats = results['statistics']
        variants.append(variant_name)
        alignment_sensitivity.append(stats['architecture_gap']['delta_alignment']['std'])
        concealment_sensitivity.append(stats['architecture_gap']['delta_concealment']['std'])
    
    x = np.arange(len(variants))
    
    # Alignment sensitivity
    ax = axes[0]
    bars = ax.bar(x, alignment_sensitivity, color='steelblue', alpha=0.7)
    ax.set_xlabel('Variant', fontsize=12)
    ax.set_ylabel('Std Dev of Δ Alignment', fontsize=12)
    ax.set_title('Architecture Sensitivity: Alignment', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, alignment_sensitivity):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    # Concealment sensitivity
    ax = axes[1]
    bars = ax.bar(x, concealment_sensitivity, color='coral', alpha=0.7)
    ax.set_xlabel('Variant', fontsize=12)
    ax.set_ylabel('Std Dev of Δ Concealment', fontsize=12)
    ax.set_title('Architecture Sensitivity: Concealment', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, concealment_sensitivity):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'architecture_sensitivity.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.close()  # Close instead of show for batch processing


def generate_insights(all_results):
    """Generate textual insights from the architecture comparison."""
    print("\n" + "="*100)
    print("KEY INSIGHTS FROM ARCHITECTURE COMPARISON")
    print("="*100)
    
    # Find which architecture is generally better
    alignment_gaps = []
    concealment_gaps = []
    
    for variant_name, results in all_results.items():
        stats = results['statistics']
        alignment_gaps.append(stats['architecture_gap']['delta_alignment']['mean'])
        concealment_gaps.append(stats['architecture_gap']['delta_concealment']['mean'])
    
    mean_alignment_gap = np.mean(alignment_gaps)
    mean_concealment_gap = np.mean(concealment_gaps)
    
    print("\n1. OVERALL ARCHITECTURE PREFERENCE:")
    if mean_alignment_gap > 0.01:
        print(f"   ✓ ViT generally maintains better alignment (avg Δ_A = +{mean_alignment_gap:.4f})")
        print("   → Global self-attention better preserves illusion structure")
    elif mean_alignment_gap < -0.01:
        print(f"   ✗ CNN generally maintains better alignment (avg Δ_A = {mean_alignment_gap:.4f})")
        print("   → Local feature processing is advantageous for these illusions")
    else:
        print(f"   ≈ No strong preference (avg Δ_A = {mean_alignment_gap:.4f})")
    
    if mean_concealment_gap > 0.01:
        print(f"   ✓ ViT generally has better concealment (avg Δ_C = +{mean_concealment_gap:.4f})")
        print("   → CNNs leak information through local artifacts")
    elif mean_concealment_gap < -0.01:
        print(f"   ✗ CNN generally has better concealment (avg Δ_C = {mean_concealment_gap:.4f})")
        print("   → ViTs may leak through global structural patterns")
    else:
        print(f"   ≈ No strong preference (avg Δ_C = {mean_concealment_gap:.4f})")
    
    # Find variant most affected by architecture choice
    print("\n2. VARIANT SENSITIVITY TO ARCHITECTURE:")
    max_gap_variant = max(all_results.items(), 
                          key=lambda x: abs(x[1]['statistics']['architecture_gap']['delta_alignment']['mean']))
    print(f"   Most affected: {max_gap_variant[0]}")
    print(f"   Δ_A = {max_gap_variant[1]['statistics']['architecture_gap']['delta_alignment']['mean']:.4f}")
    
    # Find variant with highest agreement
    print("\n3. CROSS-ARCHITECTURE AGREEMENT:")
    agreements = {name: res['statistics']['cross_architecture']['agreement']['mean'] 
                  for name, res in all_results.items()}
    best_agreement = max(agreements.items(), key=lambda x: x[1])
    worst_agreement = min(agreements.items(), key=lambda x: x[1])
    
    print(f"   Highest agreement: {best_agreement[0]} ({best_agreement[1]:.3f})")
    print(f"   Lowest agreement: {worst_agreement[0]} ({worst_agreement[1]:.3f})")
    
    if worst_agreement[1] < 0.5:
        print(f"   ⚠ Low agreement in {worst_agreement[0]} suggests architectures perceive illusion very differently")
    
    # Recommendations
    print("\n4. RECOMMENDATIONS:")
    if mean_alignment_gap > 0.02:
        print("   • Consider ViT-based evaluation as primary metric (better alignment)")
    elif mean_alignment_gap < -0.02:
        print("   • Consider CNN-based evaluation as primary metric (better alignment)")
    else:
        print("   • Both architectures show similar performance - current ViT evaluation is sufficient")
    
    if np.std(alignment_gaps) > 0.02:
        print("   • High variance across variants - architecture choice significantly affects evaluation")
        print("   • Recommend reporting both CNN and ViT scores for comprehensive evaluation")
    
    print("="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare architecture effects (CNN vs ViT) across multiple variants"
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
    print("Loading architecture comparison results...")
    all_results = load_arch_evaluation_results(args.results_dirs, args.image_size)
    
    if len(all_results) == 0:
        print("No architecture evaluation results found. Run evaluate_arch_comparison.py first.")
        return
    
    print(f"\nLoaded results for {len(all_results)} variants")
    
    # Create comparison table
    print("\n" + "="*120)
    print("ARCHITECTURE COMPARISON TABLE")
    print("="*120)
    comparison_df = create_comparison_table(all_results)
    print(comparison_df.to_string(index=False))
    print("="*120 + "\n")
    
    # Save table if output directory provided
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = output_dir / 'arch_comparison_table.csv'
        comparison_df.to_csv(csv_path, index=False)
        print(f"Comparison table saved to: {csv_path}")
    
    # Generate insights
    generate_insights(all_results)
    
    # Create plots
    print("Generating comparison plots...")
    plot_architecture_gap_comparison(all_results, args.output_dir)
    plot_cnn_vs_vit_scatter(all_results, args.output_dir)
    plot_agreement_heatmap(all_results, args.output_dir)
    plot_architecture_sensitivity(all_results, args.output_dir)


if __name__ == '__main__':
    main()