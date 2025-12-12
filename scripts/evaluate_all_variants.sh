#!/bin/bash
# evaluate_all_variants.sh
# Evaluate all generated variants and create comparisons

set -e  # Exit on error

# Define the same prompt pairs (must match generation script)
declare -a PROMPT_NAMES=(
    "dog_cat"
    "campfire_man"
    "fruit_monkey"
    "plants_marilyn"
    "kitten_puppy"
    "house_castle"
    "wine_turtle"
    "kitchen_quokka"
    "museum_camel"
    "sunflowers_vampire"
)

IMAGE_SIZE=1024

echo "========================================"
echo "Multi-Variant Evaluation & Comparison"
echo "========================================"
echo "Total prompt pairs: ${#PROMPT_NAMES[@]}"
echo "Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "========================================"

# Create comparisons directory
mkdir -p results/comparisons

# Loop through each prompt pair
for name in "${PROMPT_NAMES[@]}"; do
    echo ""
    echo "========================================="
    echo "Evaluating: $name"
    echo "========================================="
    
    # Evaluate each variant
    for variant in 0 1 2 3; do
        results_dir="results/flip.${name}.variant${variant}"
        
        if [ -d "$results_dir" ]; then
            echo ""
            echo "--- Evaluating Variant $variant ---"
            python -m visual_anagrams.evaluate \
                --results_dir "$results_dir" \
                --image_size $IMAGE_SIZE \
                --quiet
            echo "✓ Variant $variant evaluated"
        else
            echo "⚠ Warning: $results_dir not found, skipping..."
        fi
    done
    
    # Create comparison for this prompt pair
    echo ""
    echo "--- Creating Comparison ---"
    
    # Build list of existing result directories
    variant_dirs=()
    for variant in 0 1 2 3; do
        results_dir="results/flip.${name}.variant${variant}"
        if [ -d "$results_dir" ]; then
            variant_dirs+=("$results_dir")
        fi
    done
    
    # Only create comparison if we have at least 2 variants
    if [ ${#variant_dirs[@]} -ge 2 ]; then
        python -m visual_anagrams.compare_variants \
            --results_dirs "${variant_dirs[@]}" \
            --image_size $IMAGE_SIZE \
            --output_dir "results/comparisons/${name}"
        echo "✓ Comparison created: results/comparisons/${name}"
    else
        echo "⚠ Warning: Not enough variants to compare for $name"
    fi
    
    echo ""
    echo "========================================="
    echo "✓ Completed evaluation for: $name"
    echo "========================================="
done

# Create master comparison across all prompt pairs
echo ""
echo "========================================="
echo "Creating Master Comparison"
echo "========================================="

# Generate comprehensive summary statistics
echo ""
echo "--- Generating Summary Statistics ---"
python << 'END_PYTHON'
import json
import pandas as pd
from pathlib import Path

# Collect all evaluation results
all_results = []

prompt_names = [
    "dog_cat", "campfire_man", "fruit_monkey", "plants_marilyn",
    "kitten_puppy", "house_castle", "wine_turtle", "kitchen_quokka",
    "museum_camel", "sunflowers_vampire"
]

for name in prompt_names:
    for variant in [0, 1, 2, 3]:
        eval_file = Path(f"results/flip.{name}.variant{variant}/evaluation_1024.json")
        if eval_file.exists():
            with open(eval_file) as f:
                data = json.load(f)
            
            stats = data['statistics']
            all_results.append({
                'prompt_pair': name,
                'variant': f'variant{variant}',
                'n_samples': stats['num_samples'],
                'alignment_mean': stats['alignment']['mean'],
                'alignment_q90': stats['alignment']['q90'],
                'alignment_q95': stats['alignment']['q95'],
                'concealment_mean': stats['concealment']['mean'],
                'concealment_q90': stats['concealment']['q90'],
                'concealment_q95': stats['concealment']['q95'],
            })

if all_results:
    df = pd.DataFrame(all_results)
    
    # Save comprehensive results
    df.to_csv('results/comparisons/all_results.csv', index=False)
    print(f"✓ Saved comprehensive results to results/comparisons/all_results.csv")
    print(f"  Total entries: {len(df)}")
    
    # Print summary by variant
    print("\n" + "="*80)
    print("SUMMARY BY VARIANT")
    print("="*80)
    summary = df.groupby('variant').agg({
        'alignment_mean': ['mean', 'std'],
        'alignment_q90': 'mean',
        'concealment_mean': ['mean', 'std'],
        'concealment_q90': 'mean',
        'n_samples': 'sum'
    }).round(4)
    print(summary)
    
    # Save summary
    summary.to_csv('results/comparisons/summary_by_variant.csv')
    print("\n✓ Saved summary to results/comparisons/summary_by_variant.csv")
    
    # Create ranking
    print("\n" + "="*80)
    print("RANKING BY ALIGNMENT SCORE (Q90)")
    print("="*80)
    ranking = df.groupby('variant')['alignment_q90'].mean().sort_values(ascending=False)
    for i, (variant, score) in enumerate(ranking.items(), 1):
        print(f"  {i}. {variant}: {score:.4f}")
    
    print("\n" + "="*80)
    print("RANKING BY CONCEALMENT SCORE (Q90)")
    print("="*80)
    ranking = df.groupby('variant')['concealment_q90'].mean().sort_values(ascending=False)
    for i, (variant, score) in enumerate(ranking.items(), 1):
        print(f"  {i}. {variant}: {score:.4f}")
else:
    print("⚠ No evaluation results found")
END_PYTHON

echo ""
echo "========================================"
echo "✓ ALL EVALUATION COMPLETE!"
echo "========================================"
echo "Results saved to:"
echo "  - Individual comparisons: results/comparisons/<prompt_name>/"
echo "  - Master summary: results/comparisons/all_results.csv"
echo "  - Variant summary: results/comparisons/summary_by_variant.csv"
echo "========================================"