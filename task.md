# Visual Anagrams Pipeline Variants for Comparative Analysis

Based on the design considerations discussed in the paper and notebook, here are three meaningful variants to test:

---

## Variant 1: Latent Diffusion Baseline (Stable Diffusion XL)

### Rationale
The authors explicitly chose DeepFloyd IF (pixel-based) over latent diffusion models, claiming that latent representations cause artifacts because "the location of latents change, but the content and orientation of these blocks do not" (Section 3.5, Figure 3). This variant tests that core architectural claim.

### Implementation
```python
# Replace all three stages with SDXL pipeline
from diffusers import StableDiffusionXLPipeline

sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
)
sdxl_pipeline.enable_model_cpu_offload()
sdxl_pipeline = sdxl_pipeline.to(device)

# Generate directly at higher resolution (1024x1024)
# Modify sampling functions to work with SDXL's latent space
```

### Expected Insights
- **Validate artifact claims**: Verify the "thatched line" artifacts under rotation mentioned in the paper
- **Quality vs architecture tradeoff**: Determine if latent diffusion can still produce acceptable illusions despite theoretical limitations
- **Accessibility**: SDXL is more widely available and faster, making this variant more practical for deployment
- **Failure mode analysis**: Understand which transformation types are most affected by latent representation issues

---

## Variant 2: Lightweight/Accelerated Pipeline

### Rationale
The default pipeline requires high-end GPUs (V100) and uses medium-sized models. This variant explores the minimum resource requirements while maintaining illusion quality, addressing practical deployment constraints.

### Implementation
```python
# Option A: Use smallest IF models
stage_1 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-I-S-v1.0",  # Small instead of Medium
    variant="fp16",
    torch_dtype=torch.float16,
)

stage_2 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-II-S-v1.0",  # Small instead of Medium
    text_encoder=None,
    variant="fp16",
    torch_dtype=torch.float16,
)

# Option B: Skip stage 3 (upscaling) or use faster upscaler
# Option C: Reduce inference steps (test 15, 20 steps vs 30)
# Option D: Use lower guidance scale (5.0 vs 10.0)
```

### Expected Insights
- **Minimum viable quality**: Determine the smallest model size that produces recognizable illusions
- **Speed vs quality tradeoff**: Measure generation time against CLIP alignment scores
- **Inference step sensitivity**: Test if fewer denoising steps significantly hurt illusion quality (paper uses 30-100 steps)
- **Resource requirements**: Enable deployment on free-tier Colab or consumer GPUs

---

## Variant 3: Enhanced Quality Pipeline

### Rationale
The paper uses medium-sized models for balance. This variant tests whether larger models and better upscalers improve illusion quality, alignment scores, and concealment scores (metrics from Table 1-2).

### Implementation
```python
# Use largest available DeepFloyd models
stage_1 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0",  # XL instead of M
    variant="fp16",
    torch_dtype=torch.float16,
)

stage_2 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0",  # Large instead of M
    text_encoder=None,
    variant="fp16", 
    torch_dtype=torch.float16,
)

# Replace stage 3 with state-of-the-art upscaler
from diffusers import StableDiffusionUpscalePipeline
# Alternatives: Real-ESRGAN, SwinIR, BSRGAN, or latest SDXL Refiner

stage_3 = StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler",  # Or test alternatives
    torch_dtype=torch.float16
)

# Also test higher guidance scales (12.0, 15.0)
# Test more inference steps (50, 75, 100)
```

### Expected Insights
- **Quality ceiling**: Determine maximum achievable illusion quality with current architectures
- **Model size impact**: Measure improvement in alignment (A) and concealment (C) scores vs computational cost
- **Upscaler comparison**: Test if better upscalers preserve illusion properties through 4x scaling
- **Diminishing returns**: Identify point where larger models/more steps don't meaningfully improve results
- **Complex illusion feasibility**: Test if enhanced pipeline enables reliable 3-view or 4-view illusions (which authors note are "considerably more difficult")

---

## Comparative Analysis Framework

### Metrics to Track Across All Variants:
1. **Quantitative** (from paper Section 4.1):
   - Alignment score (A): `min(diag(CLIP_score_matrix))`
   - Concealment score (C): Mean trace of softmax over CLIP scores
   - A_{0.9}, A_{0.95}, C_{0.9}, C_{0.95} quantiles

2. **Practical**:
   - Generation time per illusion
   - GPU memory requirements
   - Success rate (% of attempts producing recognizable illusions)

3. **Qualitative**:
   - Visual artifact assessment (especially for Variant 1)
   - Illusion clarity and "wow factor"
   - Failure mode categorization (per Figure 9)

### Recommended Test Set:
Use the paper's datasets for reproducibility:
- **CIFAR prompts**: 45 prompt pairs from CIFAR-10 classes
- **Custom prompts**: 50 hand-curated prompt pairs from paper
- **Transformation types**: Flip, 90° rotation, 180° rotation, jigsaw, negation

This systematic comparison would provide valuable insights into the architecture-quality-efficiency tradeoffs in multi-view illusion generation.