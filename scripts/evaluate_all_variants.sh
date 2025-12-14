#!/bin/bash
# evaluate_all_variants.sh
# Evaluate all generated variants and create comparisons
export HF_HOME="/Data/shash/.cache"
export HF_HUB_CACHE="/Data/shash/.cache/hub"

set -e  # Exit on error

# Define the same prompt pairs (must match generation script)
declare -a PROMPT_NAMES=(
    "dog_cat"
    "campfire_man"
    "fruit_monkey"
    "plants_marilyn"
    "kitten_puppy"
    # "house_castle"
    # "wine_turtle"
    # "kitchen_quokka"
    # "museum_camel"
    # "sunflowers_vampire"
    # "village_horse"
    # "ship_bird"
    # "waterfalls_panda"
    # "library_deer"
    # "reef_octopus"
    # "clock_owl"
    # "garden_butterfly"
    # "lighthouse_whale"
    # "instruments_peacock"
    # "knight_dragon"
    # "coffee_fox"
    # "flowers_sloth"
    # "teddy_giraffe"
    # "table_bear"
    # "ruins_tiger"
    # "bakery_raccoon"
    # "robot_tree"
    # "telescope_rabbit"
    # "venice_swan"
    # "wizard_forest"
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
        results_dir="results/rotate_cw.${name}.variant${variant}"
        
        if [ -d "$results_dir" ]; then
            echo ""
            echo "--- Evaluating Variant $variant ---"
            
            # Check if evaluation already exists
            eval_file="${results_dir}/evaluation_${IMAGE_SIZE}.json"
            if [ -f "$eval_file" ]; then
                echo "⚠ Evaluation already exists at $eval_file, skipping..."
            else
                # Run evaluation without --quiet to see output
                python -m visual_anagrams.evaluate \
                    --results_dir "$results_dir" \
                    --image_size $IMAGE_SIZE
                
                # Verify the file was created
                if [ -f "$eval_file" ]; then
                    echo "✓ Variant $variant evaluated successfully"
                    echo "  Saved to: $eval_file"
                else
                    echo "✗ ERROR: Evaluation file not created at $eval_file"
                    exit 1
                fi
            fi
        else
            echo "⚠ Warning: $results_dir not found, skipping..."
        fi
    done
    
    # Create comparison for this prompt pair
    echo ""
    echo "--- Creating Comparison ---"
    
    # Build list of existing result directories
    variant_dirs=()
    missing_evals=0
    for variant in 0 1 2 3; do
        results_dir="results/rotate_cw.${name}.variant${variant}"
        eval_file="${results_dir}/evaluation_${IMAGE_SIZE}.json"
        
        if [ -d "$results_dir" ]; then
            if [ -f "$eval_file" ]; then
                variant_dirs+=("$results_dir")
            else
                echo "⚠ Warning: Missing evaluation file: $eval_file"
                missing_evals=$((missing_evals + 1))
            fi
        fi
    done
    
    echo "Found ${#variant_dirs[@]} variants with evaluation results"
    
    # Only create comparison if we have at least 2 variants
    if [ ${#variant_dirs[@]} -ge 2 ]; then
        python -m visual_anagrams.compare_variants \
            --results_dirs "${variant_dirs[@]}" \
            --image_size $IMAGE_SIZE \
            --output_dir "results/comparisons/${name}"
        
        if [ -f "results/comparisons/${name}/comparison_table.csv" ]; then
            echo "✓ Comparison created: results/comparisons/${name}"
            cat "results/comparisons/${name}/comparison_table.csv"
        else
            echo "⚠ Warning: Comparison files not created"
        fi
    else
        echo "⚠ Warning: Not enough variants to compare for $name (found ${#variant_dirs[@]})"
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

# Create a Python script to generate summary
cat > /tmp/generate_summary.py << 'END_PYTHON'
import json
import pandas as pd
from pathlib import Path
import sys

# Collect all evaluation results
all_results = []

prompt_names = [
    "dog_cat", "campfire_man", "fruit_monkey", "plants_marilyn",
    "kitten_puppy", "house_castle", "wine_turtle", "kitchen_quokka",
    "museum_camel", "sunflowers_vampire"
]

print("Scanning for evaluation results...")
found_count = 0
missing_count = 0

for name in prompt_names:
    for variant in [0, 1, 2, 3]:
        eval_file = Path(f"results/rotate_cw.{name}.variant{variant}/evaluation_1024.json")
        if eval_file.exists():
            try:
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
                found_count += 1
                print(f"  ✓ {name} variant{variant}")
            except Exception as e:
                print(f"  ✗ Error reading {eval_file}: {e}")
                missing_count += 1
        else:
            print(f"  ✗ Missing: {eval_file}")
            missing_count += 1

print(f"\nFound: {found_count}, Missing: {missing_count}")

if not all_results:
    print("\n⚠ ERROR: No evaluation results found!")
    sys.exit(1)

df = pd.DataFrame(all_results)

# Create output directory
Path('results/comparisons').mkdir(parents=True, exist_ok=True)

# Save comprehensive results
df.to_csv('results/comparisons/all_results.csv', index=False)
print(f"\n✓ Saved comprehensive results to results/comparisons/all_results.csv")
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
print(summary.to_string())

# Save summary
summary.to_csv('results/comparisons/summary_by_variant.csv')
print(f"\n✓ Saved summary to results/comparisons/summary_by_variant.csv")

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

# Create detailed comparison table
print("\n" + "="*80)
print("DETAILED RESULTS BY PROMPT PAIR")
print("="*80)
for name in prompt_names:
    subset = df[df['prompt_pair'] == name]
    if not subset.empty:
        print(f"\n{name}:")
        print(subset[['variant', 'alignment_mean', 'concealment_mean']].to_string(index=False))

END_PYTHON

python /tmp/generate_summary.py

echo ""
echo "========================================"
echo "✓ ALL EVALUATION COMPLETE!"
echo "========================================"
echo "Results saved to:"
echo "  - Individual comparisons: results/comparisons/<prompt_name>/"
echo "  - Master summary: results/comparisons/all_results.csv"
echo "  - Variant summary: results/comparisons/summary_by_variant.csv"
echo "========================================"

# Cleanup
rm -f /tmp/generate_summary.py