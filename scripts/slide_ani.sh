#!/bin/bash
# animate_generated.sh

# Directory where generate.py saved results
SEARCH_DIR="results"

echo "========================================"
echo "Animating Generated Visual Anagrams"
echo "Looking in: $SEARCH_DIR"
echo "========================================"

# Find all 'sample_1024.png' files recursively
find "$SEARCH_DIR" -name "sample_1024.png" | while read img_path; do
    
    # img_path looks like: results/test/flip.dog_cat.variant0/0000/sample_1024.png
    
    # 1. Get the directory containing the sample (e.g., .../0000/)
    sample_dir=$(dirname "$img_path")
    
    # 2. Get the run directory (parent of sample_dir) (e.g., .../flip.dog_cat.variant0/)
    run_dir=$(dirname "$sample_dir")
    
    # 3. Path to metadata
    meta_path="$run_dir/metadata.pkl"

    echo ""
    echo "Found Image: $img_path"

    # Check if metadata exists
    if [ ! -f "$meta_path" ]; then
        echo "❌ Skipping: metadata.pkl not found in $run_dir"
        continue
    fi

    # Optional: Check if animation already exists to save time
    # (The tool usually saves it in the run_dir or sample_dir, adjust if needed)
    # This acts as a simple check to prevent re-work
    if ls "$sample_dir"/*.mp4 1> /dev/null 2>&1; then
         echo "⏭️  Skipping: Animation already exists in $sample_dir"
         continue
    fi

    echo "🎬 Animating..."
    
    # Run the animation script
    python -m visual_anagrams.animate \
        --im_path "$img_path" \
        --metadata_path "$meta_path" || echo "⚠️ Animation failed for $img_path"

done

echo ""
echo "========================================"
echo "Batch Animation Complete"
echo "========================================"