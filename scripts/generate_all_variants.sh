#!/bin/bash
# generate_all_variants.sh
# Generate illusions for multiple prompt pairs across all variants

set -e  # Exit on error

# Define prompt pairs (prompt1, prompt2, style, short_name)
declare -a PROMPT_PAIRS=(
    "dog|cat|an oil painting of|dog_cat"
    "people at a campfire|an old man|an oil painting of|campfire_man"
    "a fruit bowl|a monkey|an oil painting of|fruit_monkey"
    "houseplants|marilyn monroe|a painting of|plants_marilyn"
    "a kitten|a puppy|a watercolor of|kitten_puppy"
    "a house|a castle|an ink drawing of|house_castle"
    "wine and cheese|a turtle|a painting of|wine_turtle"
    "a kitchen|a quokka|an oil painting of|kitchen_quokka"
    "a museum|a camel|a painting of|museum_camel"
    "sunflowers|a vampire|a painting of|sunflowers_vampire"
)



# Common generation parameters
NUM_SAMPLES=2
NUM_INFERENCE_STEPS=30
GUIDANCE_SCALE=10.0
SEED=0

echo "========================================"
echo "Multi-Variant Visual Anagrams Generation"
echo "========================================"
echo "Total prompt pairs: ${#PROMPT_PAIRS[@]}"
echo "Samples per pair: $NUM_SAMPLES"
echo "========================================"

# Loop through each prompt pair
for pair in "${PROMPT_PAIRS[@]}"; do
    IFS='|' read -r prompt1 prompt2 style name <<< "$pair"
    
    echo ""
    echo "========================================="
    echo "Generating: $name"
    echo "Prompt 1: $style $prompt1"
    echo "Prompt 2: $style $prompt2"
    echo "========================================="
    
    # Variant 0: Original DeepFloyd IF implementation
    echo ""
    echo "--- Variant 0: Original Author Implementation ---"
    python -m visual_anagrams.generate_variant0_author \
        --name "flip.${name}.variant0" \
        --prompts "$prompt1" "$prompt2" \
        --style "$style" \
        --views identity flip \
        --num_samples $NUM_SAMPLES \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --guidance_scale $GUIDANCE_SCALE \
        --seed $SEED \
        --generate_1024
    
    echo "✓ Variant 0 complete"
    
    # Variant 1: SDXL Baseline
    echo ""
    echo "--- Variant 1: SDXL Baseline ---"
    python -m visual_anagrams.generate_variant1_sdxl \
        --name "flip.${name}.variant1" \
        --prompts "$prompt1" "$prompt2" \
        --style "$style" \
        --views identity flip \
        --num_samples $NUM_SAMPLES \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --guidance_scale $GUIDANCE_SCALE \
        --seed $SEED \
        --generate_1024
    
    echo "✓ Variant 1 complete"
    
    # Variant 2: Adaptive Progressive
    echo ""
    echo "--- Variant 2: Adaptive Progressive ---"
    python -m visual_anagrams.generate_variant2_adaptive \
        --name "flip.${name}.variant2" \
        --prompts "$prompt1" "$prompt2" \
        --style "$style" \
        --views identity flip \
        --num_samples $NUM_SAMPLES \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --guidance_scale $GUIDANCE_SCALE \
        --seed $SEED \
        --generate_1024
    
    echo "✓ Variant 2 complete"
    
    # Variant 3: Frequency + Momentum
    echo ""
    echo "--- Variant 3: Frequency + Momentum ---"
    python -m visual_anagrams.generate_variant3_frequency \
        --name "flip.${name}.variant3" \
        --prompts "$prompt1" "$prompt2" \
        --style "$style" \
        --views identity flip \
        --num_samples $NUM_SAMPLES \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --guidance_scale $GUIDANCE_SCALE \
        --seed $SEED \
        --generate_1024 \
        --momentum_beta 0.9
    
    echo "✓ Variant 3 complete"
    
    echo ""
    echo "========================================="
    echo "✓ Completed all variants for: $name"
    echo "========================================="
done

echo ""
echo "========================================"
echo "✓ ALL GENERATION COMPLETE!"
echo "========================================"
echo "Generated ${#PROMPT_PAIRS[@]} prompt pairs × 4 variants = $((${#PROMPT_PAIRS[@]} * 4)) result sets"
echo "Total samples: $((${#PROMPT_PAIRS[@]} * 4 * NUM_SAMPLES))"
echo "========================================"