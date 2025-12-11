# generate_variant2_adaptive.py
"""
Variant 2: Enhanced Quality Pipeline with Progressive View Weighting

This variant implements a multi-stage adaptive approach:
1. Structure Stage: Equal view weights with high guidance to establish illusion structure
2. Balancing Stage: Adaptive weights based on CLIP scores to balance view quality
3. Refinement Stage: Progressive transition to identity-dominant for final quality

Expected to produce higher quality illusions with better balance between views.
"""

import argparse
from pathlib import Path
from PIL import Image

import torch
from diffusers import StableDiffusionXLPipeline
import torchvision.transforms.functional as TF
import clip

from visual_anagrams.views import get_views
from visual_anagrams.samplers import sample_sdxl_adaptive_progressive
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

# Load CLIP for adaptive weighting
print("Loading CLIP model for adaptive weighting...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=args.device)

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

# Save metadata
save_metadata(views, args, save_dir, variant="variant2_adaptive")

# Sample illusions
for i in range(args.num_samples):
    generator = torch.manual_seed(args.seed + i)
    sample_dir = save_dir / f'{args.seed + i:04}'
    sample_dir.mkdir(exist_ok=True, parents=True)

    height = 1024 if args.generate_1024 else 512
    width = 1024 if args.generate_1024 else 512
    
    print(f"\n{'='*60}")
    print(f"Generating sample {i+1}/{args.num_samples} at {height}x{width}")
    print(f"Using adaptive progressive sampling")
    print(f"{'='*60}\n")
    
    image = sample_sdxl_adaptive_progressive(
        sdxl_pipeline,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        views,
        prompts,  # Pass text prompts for CLIP
        height=height,
        width=width,
        ref_im=ref_im,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        reduction='adaptive',
        generator=generator,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess
    )
    
    save_illusion(image, views, sample_dir)
    print(f"✓ Saved sample {i+1} to {sample_dir}")