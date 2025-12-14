# Visual Anagrams Pipeline Variants

This directory contains three variants of the visual anagrams generation pipeline for comparative analysis, as outlined in `task.md`.

## Variants Overview

### Original Baseline (`generate.py`)
The original implementation using DeepFloyd IF Medium models:
- Stage I: IF-I-M-v1.0 (64x64)
- Stage II: IF-II-M-v1.0 (256x256)
- Stage III: SD 4x Upscaler (1024x1024)

### Variant 1: Latent Diffusion Baseline (`generate_variant1_sdxl.py`)
**Purpose**: Test the authors' claim that latent diffusion models cause artifacts.

**Implementation**:
- Replaces all three DeepFloyd IF stages with Stable Diffusion XL
- Generates directly at target resolution (512x512 or 1024x1024)
- Works in latent space instead of pixel space

**Expected Insights**:
- Validate "thatched line" artifacts under rotation (Section 3.5, Figure 3)
- Quality vs architecture tradeoff
- More accessible (SDXL is faster and more widely available)

**Usage**:
```bash
python -m visual_anagrams.generate_variant1_sdxl \
    --name sdxl.village.horse \
    --prompts "a snowy mountain village" "a horse" \
    --style "an oil painting of" \
    --views identity rotate_cw \
    --num_samples 10 \
    --num_inference_steps 30 \
    --guidance_scale 10.0 \
    --generate_1024
```

### Variant 2: Lightweight/Accelerated (`generate_variant2_lightweight.py`)
**Purpose**: Explore minimum resource requirements while maintaining quality.

**Implementation**:
- Stage I: IF-I-S-v1.0 (Small instead of Medium)
- Stage II: IF-II-S-v1.0 (Small instead of Medium)
- Stage III: Same SD 4x Upscaler

**Expected Insights**:
- Minimum viable quality threshold
- Speed vs quality tradeoff
- GPU memory requirements
- Enable deployment on free-tier Colab or consumer GPUs

**Usage**:
```bash
python -m visual_anagrams.generate_variant2_lightweight \
    --name lightweight.village.horse \
    --prompts "a snowy mountain village" "a horse" \
    --style "an oil painting of" \
    --views identity rotate_cw \
    --num_samples 10 \
    --num_inference_steps 30 \
    --guidance_scale 10.0 \
    --generate_1024
```

### Variant 3: Enhanced Quality (`generate_variant3_enhanced.py`)
**Purpose**: Test maximum achievable quality with larger models.

**Implementation**:
- Stage I: IF-I-XL-v1.0 (Extra Large instead of Medium)
- Stage II: IF-II-L-v1.0 (Large instead of Medium)
- Stage III: Same SD 4x Upscaler

**Expected Insights**:
- Quality ceiling with current architectures
- Model size impact on Alignment (A) and Concealment (C) scores
- Feasibility of complex 3-view or 4-view illusions
- Diminishing returns point

**Usage**:
```bash
python -m visual_anagrams.generate_variant3_enhanced \
    --name enhanced.village.horse \
    --prompts "a snowy mountain village" "a horse" \
    --style "an oil painting of" \
    --views identity rotate_cw \
    --num_samples 10 \
    --num_inference_steps 30 \
    --guidance_scale 10.0 \
    --generate_1024
```

## Comparative Analysis

### Metrics to Track

1. **Quantitative** (from paper Section 4.1):
   - Alignment score (A): `min(diag(CLIP_score_matrix))`
   - Concealment score (C): Mean trace of softmax over CLIP scores
   - A_{0.9}, A_{0.95}, C_{0.9}, C_{0.95} quantiles

2. **Practical**:
   - Generation time per illusion
   - GPU memory requirements
   - Success rate (% producing recognizable illusions)

3. **Qualitative**:
   - Visual artifact assessment (especially Variant 1)
   - Illusion clarity and "wow factor"
   - Failure mode categorization

### Running Comparisons

To run all variants on the same prompts:

```bash
# Original baseline
python -m visual_anagrams.generate \
    --name baseline.test \
    --prompts "a cat" "a dog" \
    --views identity rotate_cw \
    --num_samples 5

# Variant 1: SDXL
python -m visual_anagrams.generate_variant1_sdxl \
    --name sdxl.test \
    --prompts "a cat" "a dog" \
    --views identity rotate_cw \
    --num_samples 5

# Variant 2: Lightweight
python -m visual_anagrams.generate_variant2_lightweight \
    --name lightweight.test \
    --prompts "a cat" "a dog" \
    --views identity rotate_cw \
    --num_samples 5

# Variant 3: Enhanced
python -m visual_anagrams.generate_variant3_enhanced \
    --name enhanced.test \
    --prompts "a cat" "a dog" \
    --views identity rotate_cw \
    --num_samples 5
```

## Notes

- All variants save metadata with a `variant` field for tracking
- Variant 1 (SDXL) includes a custom sampler (`sample_sdxl_single_stage`) that handles latent space operations
- The same command-line arguments work across all variants
- Results are saved to `results/<name>/` directory with the same structure

## Modified Files

1. **visual_anagrams/generate_variant1_sdxl.py** - New file
2. **visual_anagrams/generate_variant2_lightweight.py** - New file
3. **visual_anagrams/generate_variant3_enhanced.py** - New file
4. **visual_anagrams/samplers.py** - Added `sample_sdxl_single_stage()` function
5. **visual_anagrams/utils.py** - Updated `save_metadata()` to accept `variant` parameter
