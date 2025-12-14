#!/bin/bash
# generate_all_variants.sh
# Generate illusions for multiple prompt pairs across all variants

export HF_HOME="/Data/shash/.cache"
export HF_HUB_CACHE="/Data/shash/.cache/hub"

set -e  # Exit on error

# Define prompt pairs (prompt1, prompt2, style, short_name)
declare -a PROMPT_PAIRS=(
    "dog|cat|a photorealistic picture of|dog_cat"
    "people at a campfire|an old man|a futuristic sytle image of|campfire_man"
    # "a fruit bowl|a monkey|an oil painting of|fruit_monkey"
    "houseplants|a photorealistic picture of|a painting of|plants_marilyn"
    # "a kitten|a puppy|a photorealistic picture of|kitten_puppy"
    # "a house|a castle|an ink drawing of|house_castle"
    # "wine and cheese|a turtle|a painting of|wine_turtle"
    # "a kitchen|a quokka|an oil painting of|kitchen_quokka"
    # "a museum|a camel|a painting of|museum_camel"
    # "sunflowers|a vampire|a painting of|sunflowers_vampire"
    # "a snowy mountain village|a horse|an oil painting of|village_horse"
    # "a ship|a bird|an oil painting of|ship_bird"
    # "waterfalls|a red panda|a painting of|waterfalls_panda"
    # "a library|a deer|an oil painting of|library_deer"
    # "a coral reef|an octopus|a watercolor of|reef_octopus"
    # "a grandfather clock|an owl|a painting of|clock_owl"
    # "a garden|a butterfly|an oil painting of|garden_butterfly"
    # "a lighthouse|a whale|a lithograph of|lighthouse_whale"
    # "musical instruments|a peacock|a painting of|instruments_peacock"
    # "a medieval knight|a dragon|an oil painting of|knight_dragon"
    # "a coffee shop|a fox|a watercolor of|coffee_fox"
    # "flower arrangements|a sloth|an oil painting of|flowers_sloth"
    # "a teddy bear|a giraffe|an oil painting of|teddy_giraffe"
    # "a dining table|a polar bear|a painting of|table_bear"
    # "ancient ruins|a tiger|a lithograph of|ruins_tiger"
    # "a bakery|a raccoon|a watercolor of|bakery_raccoon"
    # "a robot|a tree|a sketch of|robot_tree"
    # "a telescope|a rabbit|an ink drawing of|telescope_rabbit"
    # "venice canals|a swan|an oil painting of|venice_swan"
    # "a wizard|a mushroom forest|a painting of|wizard_forest"
)



# Common generation parameters
NUM_SAMPLES=4
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
    
    # echo "✓ Variant 1 complete"
    
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
    
    # echo "✓ Variant 2 complete"
    
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
    
    # echo "✓ Variant 3 complete"
    
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