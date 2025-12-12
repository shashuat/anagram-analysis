# generate_variant3_frequency_orthogonal.py (FIXED VERSION)
"""
Variant 3: Frequency-Aware Scheduling with Momentum

This variant focuses on two key innovations:

1. FREQUENCY-AWARE VIEW SCHEDULING:
   Diffusion models generate coarse-to-fine. We schedule view importance to match:
   - Early timesteps (low freq): Equal view weights for illusion structure
   - Late timesteps (high freq): Identity-dominant for final quality

2. MOMENTUM STABILIZATION:
   Use exponential moving average of noise estimates to prevent oscillation
   between competing view objectives.

Note: Earlier version had "orthogonal projection" which was incorrectly implemented.
The paper requires view TRANSFORMATIONS to be orthogonal (flip, rotate, etc.),
which they already are by construction. We don't need to project noise estimates.
"""

import argparse
from pathlib import Path
from PIL import Image

import torch
from diffusers import StableDiffusionXLPipeline
import torchvision.transforms.functional as TF

from visual_anagrams.views import get_views
from visual_anagrams.samplers import sample_sdxl_frequency_orthogonal
from visual_anagrams.utils import add_args, save_illusion, save_metadata


# Parse args
parser = argparse.ArgumentParser()
parser = add_args(parser)
# Add variant-specific args
parser.add_argument("--momentum_beta", type=float, default=0.9,
                   help="EMA coefficient for momentum (0.9 = strong momentum, 0.0 = no momentum)")
parser.add_argument("--disable_frequency_scheduling", action='store_true',
                   help="Disable frequency-aware scheduling (for ablation)")
args = parser.parse_args()

# Do admin stuff
save_dir = Path(args.save_dir) / args.name
save_dir.mkdir(exist_ok=True, parents=True)

# Load reference image (for inverse problems)
if args.ref_im_path is not None:
    ref_im = Image.open(args.ref_im_path)
    ref_im = TF.to_tensor(ref_im) * 2 - 1
else:
    ref_im = None

# Load Stable Diffusion XL pipeline
print("Loading Stable Diffusion XL pipeline...")
sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
sdxl_pipeline = sdxl_pipeline.to(args.device)

# Ensure VAE is configured properly
if hasattr(sdxl_pipeline.vae.config, 'force_upcast'):
    sdxl_pipeline.vae.config.force_upcast = True

# Get prompts
prompts = [f'{args.style} {p}'.strip() for p in args.prompts]

# Get prompt embeddings
prompt_embeds_list = []
pooled_prompt_embeds_list = []
negative_prompt_embeds_list = []
negative_pooled_prompt_embeds_list = []

for prompt in prompts:
    (pos_embeds, 
     neg_embeds, 
     pos_pooled, 
     neg_pooled) = sdxl_pipeline.encode_prompt(
        prompt=prompt,
        device=args.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True
    )
    prompt_embeds_list.append(pos_embeds)
    pooled_prompt_embeds_list.append(pos_pooled)
    negative_prompt_embeds_list.append(neg_embeds)
    negative_pooled_prompt_embeds_list.append(neg_pooled)

# Stack embeddings
prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
pooled_prompt_embeds = torch.cat(pooled_prompt_embeds_list, dim=0)
negative_prompt_embeds = torch.cat(negative_prompt_embeds_list, dim=0)
negative_pooled_prompt_embeds = torch.cat(negative_pooled_prompt_embeds_list, dim=0)

print(f"Prompt embeddings shape: {prompt_embeds.shape}")

# Get views and validate compatibility
views = get_views(args.views, view_args=args.view_args)

print("Validating views for SDXL compatibility...")
for i, view in enumerate(views):
    view_class_name = view.__class__.__name__
    
    if 'Permute' in view_class_name or view_class_name in [
        'InnerCircleView', 'InnerRotationView', 'JigsawView', 
        'SquareHingeView', 'SkewView'
    ]:
        raise ValueError(
            f"\n{'='*60}\n"
            f"ERROR: View '{view_class_name}' is incompatible with SDXL.\n"
            f"{'='*60}\n"
            f"Compatible views: identity, flip, rotate_cw, rotate_ccw, negate\n"
            f"{'='*60}\n"
        )

print(f"✓ All {len(views)} views are compatible")

# Print variant configuration
print("\n" + "="*60)
print("VARIANT 3: Frequency + Momentum Configuration")
print("="*60)
print(f"Frequency Scheduling: {not args.disable_frequency_scheduling}")
print(f"Momentum Beta: {args.momentum_beta}")
print("="*60 + "\n")

# Save metadata
metadata_extra = {
    'variant': 'variant3_frequency_momentum',
    'momentum_beta': args.momentum_beta,
    'frequency_scheduling': not args.disable_frequency_scheduling,
}
save_metadata(views, args, save_dir, **metadata_extra)

# Sample illusions
for i in range(args.num_samples):
    generator = torch.manual_seed(args.seed + i)
    sample_dir = save_dir / f'{args.seed + i:04}'
    sample_dir.mkdir(exist_ok=True, parents=True)

    height = 1024 if args.generate_1024 else 512
    width = 1024 if args.generate_1024 else 512
    
    print(f"\n{'='*60}")
    print(f"Generating sample {i+1}/{args.num_samples} at {height}x{width}")
    print(f"Seed: {args.seed + i}")
    print(f"{'='*60}\n")
    
    image = sample_sdxl_frequency_orthogonal(
        sdxl_pipeline,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        views,
        height=height,
        width=width,
        ref_im=ref_im,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        momentum_beta=args.momentum_beta,
        orthogonal_projection=False,  # DISABLED - was incorrectly implemented
        frequency_scheduling=not args.disable_frequency_scheduling,
    )
    
    save_illusion(image, views, sample_dir)
    print(f"✓ Saved sample {i+1} to {sample_dir}")

print("\n" + "="*60)
print("Generation complete!")
print("="*60)