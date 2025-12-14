#!/bin/bash
# evaluate_all_arch.sh
# 
# Batch script to evaluate all variants with CNN vs ViT architecture comparison
# 
# Usage: bash evaluate_all_arch.sh

export HF_HOME="/Data/shash/.cache"
export HF_HUB_CACHE="/Data/shash/.cache/hub"
export TORCH_HOME="/Data/shash/.cache/torch"
export XDG_CACHE_HOME="/Data/shash/.cache"


set -e  # Exit on error

echo "=================================="
echo "Architecture Comparison Evaluation"
echo "=================================="
echo ""

# Configuration
PROMPT_PAIRS=(
    "jigsaw.dog_cat"
    "jigsaw.campfire_man"
    "jigsaw.fruit_monkey"
    "jigsaw.plants_marilyn"
    "jigsaw.kitten_puppy"
)

VARIANTS=(0 1 2 3)
IMAGE_SIZE=1024

# Check if results directory exists
if [ ! -d "results" ]; then
    echo "❌ ERROR: results/ directory not found"
    echo "   Please generate some illusions first"
    exit 1
fi

# Count total evaluations
total_evals=$((${#PROMPT_PAIRS[@]} * ${#VARIANTS[@]}))
current_eval=0

echo "Will evaluate ${total_evals} variant-prompt combinations"
echo ""

# Evaluate each combination
for pair in "${PROMPT_PAIRS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        current_eval=$((current_eval + 1))
        results_dir="results/${pair}.variant${variant}"
        
        # Check if directory exists
        if [ ! -d "$results_dir" ]; then
            echo "⚠️  [${current_eval}/${total_evals}] Skipping ${pair}.variant${variant} (not found)"
            continue
        fi
        
        # Check if already evaluated
        if [ -f "${results_dir}/evaluation_arch_${IMAGE_SIZE}.json" ]; then
            echo "✓  [${current_eval}/${total_evals}] Already evaluated ${pair}.variant${variant}"
            continue
        fi
        
        echo "▶  [${current_eval}/${total_evals}] Evaluating ${pair}.variant${variant}..."
        
        # Run architecture comparison evaluation
        python -m visual_anagrams.evaluate_arch_comparison \
            --results_dir "$results_dir" \
            --image_size "$IMAGE_SIZE" || {
                echo "❌ ERROR evaluating ${pair}.variant${variant}"
                continue
            }
        
        echo "✓  [${current_eval}/${total_evals}] Completed ${pair}.variant${variant}"
        echo ""
    done
done

echo ""
echo "=================================="
echo "Individual Evaluations Complete!"
echo "=================================="
echo ""

# Now run comparisons for each prompt pair
echo "=================================="
echo "Running Cross-Variant Comparisons"
echo "=================================="
echo ""

for pair in "${PROMPT_PAIRS[@]}"; do
    echo "▶  Comparing variants for ${pair}..."
    
    # Build list of result directories for this prompt pair
    result_dirs=()
    for variant in "${VARIANTS[@]}"; do
        results_dir="results/${pair}.variant${variant}"
        if [ -f "${results_dir}/evaluation_arch_${IMAGE_SIZE}.json" ]; then
            result_dirs+=("$results_dir")
        fi
    done
    
    # Only run comparison if we have at least 2 variants
    if [ ${#result_dirs[@]} -lt 2 ]; then
        echo "⚠️  Skipping ${pair} (less than 2 variants evaluated)"
        echo ""
        continue
    fi
    
    # Run comparison
    output_dir="results/arch_comparisons/${pair}"
    mkdir -p "$output_dir"
    
    python -m visual_anagrams.compare_arch_variants \
        --results_dirs "${result_dirs[@]}" \
        --image_size "$IMAGE_SIZE" \
        --output_dir "$output_dir" || {
            echo "❌ ERROR comparing ${pair}"
            continue
        }
    
    echo "✓  Comparison saved to ${output_dir}"
    echo ""
done

echo ""
echo "=================================="
echo "All Architecture Evaluations Complete!"
echo "=================================="
echo ""
echo "Results saved to:"
echo "  - Individual: results/<prompt_pair>.variant<N>/evaluation_arch_1024.json"
echo "  - Comparisons: results/arch_comparisons/<prompt_pair>/"
echo ""
echo "Next steps:"
echo "  1. Review individual results: cat results/flip.dog_cat.variant3/evaluation_arch_1024.json"
echo "  2. View comparison plots: results/arch_comparisons/flip.dog_cat/*.png"
echo "  3. Analyze CSV data: results/arch_comparisons/flip.dog_cat/arch_comparison_table.csv"
echo ""