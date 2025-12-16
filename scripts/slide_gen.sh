#!/bin/bash
# generate_all_views.sh

export HF_HOME="/Data/shash/.cache"
export HF_HUB_CACHE="/Data/shash/.cache/hub"

# We turn off 'set -e' so the script continues even if one generation fails
set +e 

# 1. Define Prompt Pairs (prompt1|prompt2|style|short_name)
declare -a PROMPT_PAIRS=(
    # "dog|cat|an oil painting of|dog_cat"
    "people at a campfire|an old man|an oil painting of|campfire_man"
    "a snowy mountain village|a horse|an oil painting of|village_horse"
    "a ship|a bird|an oil painting of|ship_bird"

    # "a medieval knight|a dragon|an oil painting of|knight_dragon"
    # "a coffee shop|a fox|a watercolor of|coffee_fox"
    # Add your other prompts here
)


# 2. Define ALL Views you want to attempt
declare -a VIEWS=(
    "flip"
    "inner_circle"
    "jigsaw"
    "patch_permute"
    "pixel_permute"
    "rotate_ccw"
    "rotate_cw"
    "skew"
    "square_hinge"
)

# 3. Define SDXL Compatible Views (for Variants 1, 2, 3)
declare -a SDXL_VIEWS=("flip" "rotate_cw" "rotate_ccw" "negate")

# Common parameters
NUM_SAMPLES=1
NUM_INFERENCE_STEPS=30
GUIDANCE_SCALE=10.0
SEED=0
SAVE_DIR="results/test"

echo "========================================"
echo "Smart Multi-View Generation (Pre-filtered)"
echo "========================================"

for pair in "${PROMPT_PAIRS[@]}"; do
    IFS='|' read -r prompt1 prompt2 style name <<< "$pair"

    for view_type in "${VIEWS[@]}"; do
        
        echo ""
        echo "Processing: $name | View: $view_type"

        for variant_idx in 0 1 2 3; do
            
            # --- PRE-CHECK COMPATIBILITY ---
            # If variant is 1, 2, or 3 (SDXL), check if view is allowed
            if [[ "$variant_idx" -ne 0 ]]; then
                is_compatible=0
                for allowed in "${SDXL_VIEWS[@]}"; do
                    if [[ "$view_type" == "$allowed" ]]; then
                        is_compatible=1
                        break
                    fi
                done

                if [[ "$is_compatible" -eq 0 ]]; then
                    # Skip silently or with a small log to save screen space
                    # echo "  [Skip] Variant $variant_idx does not support $view_type"
                    continue 
                fi
            fi
            # -------------------------------

            case $variant_idx in
                0) script="visual_anagrams.generate_variant0_author"; variant_name="variant0" ;;
                1) script="visual_anagrams.generate_variant1_sdxl"; variant_name="variant1" ;;
                2) script="visual_anagrams.generate_variant2_adaptive"; variant_name="variant2" ;;
                3) script="visual_anagrams.generate_variant3_frequency"; variant_name="variant3" ;;
            esac

            full_run_name="${view_type}.${name}.${variant_name}"
            
            echo "  > Running $variant_name..."

            python -m "$script" \
                --name "$full_run_name" \
                --prompts "$prompt1" "$prompt2" \
                --style "$style" \
                --views identity "$view_type" \
                --num_samples $NUM_SAMPLES \
                --num_inference_steps $NUM_INFERENCE_STEPS \
                --guidance_scale $GUIDANCE_SCALE \
                --seed $SEED \
                --generate_1024 \
                --save_dir "$SAVE_DIR" || echo "    ⚠️ Error running $full_run_name"
            
        done
    done
done