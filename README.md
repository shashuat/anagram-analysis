# Visual Anagrams Analysis

**Author:** [@shashuat](https://github.com/shashuat)

A comprehensive research project for analyzing and comparing different architectural approaches to generating multi-view optical illusions (visual anagrams) using diffusion models. This project extends the original [Visual Anagrams](https://dangeng.github.io/visual_anagrams/) (CVPR 2024 Oral) work by Daniel Geng, Aaron Park, and Andrew Owens.

## 🎯 Project Overview

Visual anagrams are images that change appearance or identity when transformed—for example, an image might appear as a "horse" normally but reveal a "village" when rotated 90 degrees. This project implements and compares multiple variant approaches to generating these optical illusions, with a focus on:

1. **Architecture Comparison**: Testing CNN (ResNet) vs ViT (Vision Transformer) biases in perceiving illusions
2. **Generation Method Variants**: Comparing DeepFloyd IF (pixel-space) vs Stable Diffusion XL (latent-space) approaches
3. **Quality Enhancement**: Exploring adaptive weighting and frequency-aware scheduling techniques
4. **Quantitative Evaluation**: Comprehensive metrics for alignment, concealment, and cross-architecture agreement

## ✨ Key Features

- **Multiple Generation Variants** (4 variants total):
  - **Variant 0**: Original DeepFloyd IF implementation (pixel-space diffusion)
  - **Variant 1**: Stable Diffusion XL baseline (latent-space diffusion)
  - **Variant 2**: Adaptive progressive weighting approach
  - **Variant 3**: Frequency-aware scheduling with momentum stabilization

- **Architecture-Aware Evaluation**:
  - Dual evaluation using CNN (ResNet-50) and ViT (Vision Transformer) encoders
  - Measures how architectural biases affect illusion perception
  - Cross-architecture agreement and correlation metrics

- **Comprehensive Metrics**:
  - **Alignment Score (A)**: How well each view matches its intended prompt
  - **Concealment Score (C)**: How well alternative views are hidden
  - **Architecture Gap (Δ)**: Performance difference between CNN and ViT
  - Statistical analysis with quantiles (90th, 95th percentiles)

- **Supported Transformations**:
  - Rotation (90°, 180°, 270°)
  - Flip (horizontal/vertical)
  - Inner circle rotation
  - Jigsaw rearrangement
  - Color inversion
  - Patch/pixel permutation
  - And more...

## 📋 Requirements

### System Requirements
- Python ≥ 3.11
- CUDA-capable GPU (16GB+ VRAM recommended for SDXL variants)
- ~50GB disk space for models and cache

### Dependencies

Core dependencies include:
- **Diffusion Models**: `diffusers`, `transformers`, `accelerate`
- **Deep Learning**: `torch`, `torchvision`
- **Vision-Language**: `clip` (via OpenAI)
- **Utilities**: `Pillow`, `numpy`, `wandb`, `python-dotenv`

See [requirements.txt](requirements.txt) or [pyproject.toml](pyproject.toml) for complete list.

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/shashuat/anagram-analysis.git
cd anagram-analysis
```

### 2. Create Python Environment

Using conda (recommended):
```bash
conda create -n anagram-analysis python=3.11
conda activate anagram-analysis
```

Or using venv:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or using uv (faster):
```bash
uv pip install -r requirements.txt
```

### 4. Setup DeepFloyd IF Access

DeepFloyd IF requires accepting usage conditions:

1. Create a [Hugging Face account](https://huggingface.co/join)
2. Accept the license at [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)
3. Create a [Hugging Face access token](https://huggingface.co/docs/hub/security-tokens)
4. Login locally:

```bash
huggingface-cli login
```

### 5. Configure Cache Paths (Optional)

Set environment variables for cache locations:

```bash
export HF_HOME="/path/to/your/.cache"
export HF_HUB_CACHE="/path/to/your/.cache/hub"
export TORCH_HOME="/path/to/your/.cache/torch"
```

Or create a `.env` file in the project root.

## 📖 Usage

### Quick Start: Generate Your First Illusion

Generate a rotation illusion (horse ↔ village):

```bash
python -m visual_anagrams.generate_variant0_author \
    --name rotate_cw.village.horse \
    --prompts "a snowy mountain village" "a horse" \
    --style "an oil painting of" \
    --views identity rotate_cw \
    --num_samples 10 \
    --num_inference_steps 30 \
    --guidance_scale 10.0 \
    --generate_1024
```

Results will be saved to `results/rotate_cw.village.horse/`

### Batch Generation with Scripts

Generate multiple illusions across variants:

```bash
# Edit scripts/generate_all_variants.sh to configure prompt pairs
bash scripts/generate_all_variants.sh
```

### Evaluation

Evaluate generated illusions:

```bash
# Standard CLIP evaluation (ViT)
python -m visual_anagrams.evaluate \
    --results_dir results/rotate_cw.village.horse.variant0 \
    --image_size 1024

# Architecture comparison (CNN vs ViT)
python -m visual_anagrams.evaluate_arch_comparison \
    --results_dir results/rotate_cw.village.horse.variant0 \
    --image_size 1024

# Batch evaluation
bash scripts/evaluate_all_arch.sh
```

### Compare Variants

```bash
python -m visual_anagrams.compare_arch_variants \
    --results_dirs results/inner_circle.dog_cat.variant0 \
                   results/inner_circle.dog_cat.variant1 \
                   results/inner_circle.dog_cat.variant2 \
                   results/inner_circle.dog_cat.variant3 \
    --image_size 1024 \
    --output_dir results_arch_flip/arch_comparisons
```

### Animation

Create animated visualizations of transformations:

```bash
python -m visual_anagrams.animate \
    --im_path results/rotate_cw.village.horse/0000/sample_1024.png \
    --metadata_path results/rotate_cw.village.horse/metadata.pkl
```

## 🔬 Generation Variants

### Variant 0: Original DeepFloyd IF (Baseline)

**Purpose**: Original pixel-space diffusion approach from the paper

**Architecture**:
- Stage I: DeepFloyd IF-I-M-v1.0 (64×64)
- Stage II: DeepFloyd IF-II-M-v1.0 (256×256)
- Stage III: SD x4 Upscaler (1024×1024)

**Usage**:
```bash
python -m visual_anagrams.generate_variant0_author \
    --name inner_circle.dog_cat.variant0 \
    --prompts "a dog" "a cat" \
    --style "a photorealistic picture of" \
    --views identity inner_circle \
    --num_samples 4
```

### Variant 1: Stable Diffusion XL

**Purpose**: Test latent diffusion artifacts hypothesis

The original paper claims latent diffusion models produce artifacts because "the location of latents change, but the content and orientation of these blocks do not" (Section 3.5). This variant tests this claim empirically.

**Architecture**: Single-stage SDXL (latent-space diffusion)

**Usage**:
```bash
python -m visual_anagrams.generate_variant1_sdxl \
    --name flip.dog_cat.variant1 \
    --prompts "a dog" "a cat" \
    --style "a photorealistic picture of" \
    --views identity flip \
    --num_samples 4
```

**Notes**: 
- Only works with spatial transformations (rotate, flip)
- Not compatible with permutation-based views (jigsaw, inner_circle, etc.)
- Faster generation but may show "thatched line" artifacts

### Variant 2: Adaptive Progressive Weighting

**Purpose**: Enhanced quality through dynamic view balancing

**Key Innovation**: Three-stage adaptive approach:
1. **Structure Stage**: Equal view weights to establish illusion
2. **Balancing Stage**: CLIP-guided adaptive weights
3. **Refinement Stage**: Identity-dominant for final quality

**Usage**:
```bash
python -m visual_anagrams.generate_variant2_adaptive \
    --name flip.dog_cat.variant2 \
    --prompts "a dog" "a cat" \
    --style "a photorealistic picture of" \
    --views identity flip \
    --num_samples 4
```

### Variant 3: Frequency-Aware Scheduling

**Purpose**: Match view scheduling to diffusion's coarse-to-fine generation

**Key Innovations**:
1. **Frequency-Aware Scheduling**: Equal weights early (structure), identity-dominant late (detail)
2. **Momentum Stabilization**: EMA of noise estimates to prevent oscillation

**Usage**:
```bash
python -m visual_anagrams.generate_variant3_frequency \
    --name flip.dog_cat.variant3 \
    --prompts "a dog" "a cat" \
    --style "a photorealistic picture of" \
    --views identity flip \
    --num_samples 4 \
    --momentum_beta 0.9
```

## 📊 Evaluation Metrics

### Alignment Score (A)

Measures how well each view matches its intended prompt:

$$A = \min_{i} S_{ii}$$

where $S_{ij}$ is the CLIP similarity between view $i$ and prompt $j$.

- Higher is better (max = 1.0)
- Measures worst-case view alignment
- Reported as mean, 90th percentile (A₀.₉), 95th percentile (A₀.₉₅)

### Concealment Score (C)

Measures how well alternative interpretations are hidden:

$$C = \frac{1}{N} \text{tr}(\text{softmax}(S/\tau))$$

where $\tau$ is a temperature parameter.

- Higher is better (max = 1.0)
- Measures view disambiguation
- Closer to 1.0 means views are well-concealed

### Architecture Gap (Δ)

Difference between ViT and CNN performance:

$$\Delta_A = A_{\text{ViT}} - A_{\text{CNN}}$$
$$\Delta_C = C_{\text{ViT}} - C_{\text{CNN}}$$

- Positive: ViT performs better
- Negative: CNN performs better
- Measures architectural bias impact

### Cross-Architecture Agreement

Measures how similarly CNN and ViT perceive the illusion:
- **Agreement**: Percentage of samples where both architectures agree on best view
- **Correlation**: Pearson correlation of score matrices

## 🗂️ Project Structure

```
anagram-analysis/
├── visual_anagrams/           # Main package
│   ├── generate_variant0_author.py    # Original DeepFloyd IF
│   ├── generate_variant1_sdxl.py      # SDXL baseline
│   ├── generate_variant2_adaptive.py  # Adaptive weighting
│   ├── generate_variant3_frequency.py # Frequency-aware
│   ├── evaluate.py                    # Standard evaluation
│   ├── evaluate_arch_comparison.py    # CNN vs ViT evaluation
│   ├── compare_variants.py            # Cross-variant comparison
│   ├── compare_arch_variants.py       # Architecture comparison
│   ├── animate.py                     # Create animations
│   ├── samplers.py                    # Sampling algorithms
│   ├── utils.py                       # Utilities
│   └── views/                         # Transformation definitions
├── scripts/                   # Batch processing scripts
│   ├── generate_all_variants.sh       # Batch generation
│   ├── evaluate_all_arch.sh           # Batch evaluation
│   └── evaluate_all_variants.sh       # Standard batch eval
├── results/                   # Generated illusions (organized by experiment)
├── results_cw/               # Rotation experiments
├── results_flip/             # Flip experiments
├── results_arch_flip/        # Architecture comparison results
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project configuration
├── LICENSE                  # Apache 2.0 license
└── README.md               # This file
```

## 🎨 Tips for Choosing Prompts

Based on experimentation and the original paper's guidance:

### ✅ What Works Well

- **Artistic styles**: "an oil painting of", "a watercolor of", "a sketch of"
  - More flexibility in interpretation than photorealism
  
- **Faces for hidden views**: "an old man", "marilyn monroe"
  - Human visual system excels at face detection
  
- **Flexible subjects**: "houseplants", "a kitchen", "wine and cheese"
  - Can be depicted in many ways while remaining recognizable

### ❌ What's Challenging

- **Photorealism**: "a photo of" constraints are difficult
  - Still possible but requires more trials
  
- **Abstract concepts**: Hard to recognize instantly
  
- **3-4 view illusions**: Significantly harder than 2-view
  - Need very compatible prompts

### 💡 General Advice

- **Exploration is key**: Intuition often misleads—test widely
- **Instant recognition**: Best illusions are immediately clear
- **Pair complementary subjects**: Consider visual compatibility

## 📚 Example Results

The `results/` directories contain generated illusions organized by:
- **Transformation type**: `rotate_cw`, `flip`, `inner_circle`, `jigsaw`, etc.
- **Prompt pair**: `dog_cat`, `campfire_man`, etc.
- **Variant**: `variant0`, `variant1`, `variant2`, `variant3`

Each result directory includes:
- `0000/`, `0001/`, ... : Individual samples
  - `sample_64.png` : 64×64 image
  - `sample_256.png` : 256×256 image
  - `sample_1024.png` : 1024×1024 image (if `--generate_1024` used)
- `metadata.pkl` : Generation parameters and view information
- `evaluation_1024.json` : CLIP evaluation results (standard)
- `evaluation_arch_1024.json` : Architecture comparison results

## 🔧 Advanced Configuration

### Custom Transformations

Views are defined in `visual_anagrams/views/`. To create custom transformations:

1. Inherit from `BaseView`
2. Implement the `view()` method
3. For permutation-based views, use `PermuteView` class

See existing views for examples.

### Memory Optimization

For limited GPU memory:

```bash
# Use smaller batch sizes
--num_samples 1

# Skip 1024×1024 generation
# (omit --generate_1024 flag)

# Use CPU offloading (automatically enabled)
# Models are moved to CPU when not in use
```

### Hyperparameter Tuning

Key parameters to adjust:

- `--num_inference_steps`: More steps = higher quality but slower (default: 30)
- `--guidance_scale`: Higher = stronger prompt adherence (default: 10.0)
- `--seed`: For reproducibility
- `--momentum_beta`: (Variant 3 only) EMA coefficient (default: 0.9)

## 📈 Experimental Results

Results from architecture comparison experiments are saved in:
- `results_cw/comparisons/` : Rotation experiments analysis
- `results_flip/comparisons/` : Flip experiments analysis
- `results_arch_flip/arch_comparisons/` : Cross-architecture analysis

Generated plots include:
- Score distributions (alignment and concealment)
- Architecture gap analysis
- Cross-architecture agreement heatmaps
- Per-sample comparisons

## 🤝 Contributing

This is a research project. Contributions, suggestions, and discussions are welcome!

### Areas for Exploration

- Additional transformation types
- Alternative architectures (other diffusion models)
- Novel evaluation metrics
- Three-view and four-view illusion techniques
- Quality improvements for specific transformation types

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project builds upon the original [Visual Anagrams](https://dangeng.github.io/visual_anagrams/) work:

```
@inproceedings{geng2024visual,
  title={Visual Anagrams: Generating Multi-View Optical Illusions with Diffusion Models},
  author={Geng, Daniel and Park, Aaron and Owens, Andrew},
  booktitle={CVPR},
  year={2024}
}
```

**Original Paper**: [Arxiv](https://arxiv.org/abs/2311.17919) | [Website](https://dangeng.github.io/visual_anagrams/) | [Code](https://github.com/dangeng/visual_anagrams)

Thanks to:
- Daniel Geng, Aaron Park, and Andrew Owens for the original implementation and paper
- The Hugging Face team for `diffusers` and model hosting
- OpenAI for CLIP

## 📞 Contact

**Author**: [@shashuat](https://github.com/shashuat)

For questions, issues, or collaboration inquiries, please open an issue on GitHub.

---

<div align="center">

**Happy Illusion Crafting! 🎨✨**

</div>