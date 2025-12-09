"""
Variant 1: Latent Diffusion Baseline (Stable Diffusion XL)

This variant replaces the three-stage DeepFloyd IF pipeline with Stable Diffusion XL,
testing the authors' claim that latent diffusion models cause artifacts due to 
"the location of latents change, but the content and orientation of these blocks do not"
(Section 3.5, Figure 3).

Expected to show "thatched line" artifacts under rotation mentioned in the paper.
"""

import argparse
from pathlib import Path
from PIL import Image

import torch
from diffusers import StableDiffusionXLPipeline
import torchvision.transforms.functional as TF

from visual_anagrams.views import get_views
from visual_anagrams.samplers import sample_sdxl_single_stage
from visual_anagrams.utils import add_args, save_illusion, save_metadata


# Parse args
parser = argparse.ArgumentParser()
parser = add_args(parser)
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
# sdxl_pipeline.enable_model_cpu_offload()
sdxl_pipeline = sdxl_pipeline.to(args.device)

# Ensure VAE is configured for upcast to avoid NaN issues
if hasattr(sdxl_pipeline.vae.config, 'force_upcast'):
    sdxl_pipeline.vae.config.force_upcast = True

# Get prompts
prompts = [f'{args.style} {p}'.strip() for p in args.prompts]

# Get prompt embeddings
prompt_embeds_list = []
pooled_prompt_embeds_list = []
negative_prompt_embeds_list = []
negative_pooled_prompt_embeds_list = []

for idx, prompt in enumerate(prompts):
    # SDXL has two text encoders, we need to get embeddings from both
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

# Stack embeddings for all views
prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
pooled_prompt_embeds = torch.cat(pooled_prompt_embeds_list, dim=0)
negative_prompt_embeds = torch.cat(negative_prompt_embeds_list, dim=0)
negative_pooled_prompt_embeds = torch.cat(negative_pooled_prompt_embeds_list, dim=0)

print(f"Prompt embeddings shape: {prompt_embeds.shape}")
print(f"Pooled prompt embeddings shape: {pooled_prompt_embeds.shape}")

# Get views
views = get_views(args.views, view_args=args.view_args)

# Validate views are compatible with latent space diffusion
print("Validating views for SDXL compatibility...")
for i, view in enumerate(views):
    view_class_name = view.__class__.__name__
    
    # Check if it's a permutation-based view that won't work in latent space
    if 'Permute' in view_class_name or view_class_name in [
        'InnerCircleView', 'InnerRotationView', 'JigsawView', 
        'SquareHingeView', 'SkewView'
    ]:
        raise ValueError(
            f"\n{'='*60}\n"
            f"ERROR: View '{view_class_name}' is incompatible with SDXL.\n"
            f"{'='*60}\n"
            f"SDXL works in latent space (128×128) not pixel space (1024×1024).\n"
            f"Permutation-based views require pixel-level operations.\n\n"
            f"Compatible views: identity, flip, rotate_cw, rotate_ccw, negate\n"
            f"Incompatible views: inner_circle, jigsaw, patch_permute, \n"
            f"                   pixel_permute, square_hinge, skew\n"
            f"{'='*60}\n"
        )

print(f"✓ All {len(views)} views are compatible with SDXL")

# Save metadata
save_metadata(views, args, save_dir, variant="variant1_sdxl")

# Sample illusions
for i in range(args.num_samples):
    # Admin stuff
    generator = torch.manual_seed(args.seed + i)
    sample_dir = save_dir / f'{args.seed + i:04}'
    sample_dir.mkdir(exist_ok=True, parents=True)

    # Generate image directly at target resolution (1024x1024 or 512x512)
    # SDXL works in latent space, generating directly at high resolution
    height = 1024 if args.generate_1024 else 512
    width = 1024 if args.generate_1024 else 512
    
    print(f"Generating sample {i+1}/{args.num_samples} at {height}x{width}...")
    
    image = sample_sdxl_single_stage(
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
        reduction=args.reduction,
        generator=generator
    )
    
    save_illusion(image, views, sample_dir)
    print(f"Saved sample {i+1} to {sample_dir}")
