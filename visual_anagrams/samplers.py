# samplers.py -- Sampling functions for multi-view diffusion models

from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from diffusers.utils.torch_utils import randn_tensor

@torch.no_grad()
def sample_stage_1(model,
                   prompt_embeds,
                   negative_prompt_embeds, 
                   views,
                   ref_im=None,
                   num_inference_steps=100,
                   guidance_scale=7.0,
                   reduction='mean',
                   generator=None):

    # Params
    num_images_per_prompt = 1
    #device = model.device
    device = torch.device('cuda')   # Sometimes model device is cpu???
    height = model.unet.config.sample_size
    width = model.unet.config.sample_size
    batch_size = 1      # TODO: Support larger batch sizes, maybe
    num_prompts = prompt_embeds.shape[0]
    assert num_prompts == len(views), \
        "Number of prompts must match number of views!"

    # For CFG
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # Setup timesteps
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = model.scheduler.timesteps

    # Make intermediate_images
    noisy_images = model.prepare_intermediate_images(
        batch_size * num_images_per_prompt,
        model.unet.config.in_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
    )

    # Resize ref image to correct size
    if ref_im is not None:
        ref_im = TF.resize(ref_im, height)
        ref_im = ref_im.to(noisy_images.device).to(noisy_images.dtype)

    for i, t in enumerate(tqdm(timesteps)):
        # If solving an inverse problem, then project x_t so
        # that first component matches reference image's first component
        if ref_im is not None:
            # Inject noise to reference image
            alpha_cumprod = model.scheduler.alphas_cumprod[t]
            ref_noisy = torch.sqrt(alpha_cumprod) * ref_im + \
                        torch.sqrt(1 - alpha_cumprod) * torch.randn_like(ref_im)

            # Replace 1st component of x_t with 1st component of noisy ref image
            ref_noisy_component = views[0].inverse_view(ref_noisy)
            noisy_images_component = views[1].inverse_view(noisy_images[0])
            noisy_images = ref_noisy_component + noisy_images_component

            # Add back batch dim
            noisy_images = noisy_images[None]

        # Apply views to noisy_image
        viewed_noisy_images = []
        for view_fn in views:
            viewed_noisy_images.append(view_fn.view(noisy_images[0]))
        viewed_noisy_images = torch.stack(viewed_noisy_images)

        # Duplicate inputs for CFG
        # Model input is: [ neg_0, neg_1, ..., pos_0, pos_1, ... ]
        model_input = torch.cat([viewed_noisy_images] * 2)
        model_input = model.scheduler.scale_model_input(model_input, t)

        # Predict noise estimate
        noise_pred = model.unet(
            model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=None,
            return_dict=False,
        )[0]

        # Extract uncond (neg) and cond noise estimates
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        # Invert the unconditional (negative) estimates
        inverted_preds = []
        for pred, view in zip(noise_pred_uncond, views):
            inverted_pred = view.inverse_view(pred)
            inverted_preds.append(inverted_pred)
        noise_pred_uncond = torch.stack(inverted_preds)

        # Invert the conditional estimates
        inverted_preds = []
        for pred, view in zip(noise_pred_text, views):
            inverted_pred = view.inverse_view(pred)
            inverted_preds.append(inverted_pred)
        noise_pred_text = torch.stack(inverted_preds)

        # Split into noise estimate and variance estimates
        noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
        noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Reduce predicted noise and variances
        noise_pred = noise_pred.view(-1,num_prompts,3,64,64)
        predicted_variance = predicted_variance.view(-1,num_prompts,3,64,64)
        if reduction == 'mean':
            noise_pred = noise_pred.mean(1)
            predicted_variance = predicted_variance.mean(1)
        elif reduction == 'sum':
            # For factorized diffusion
            noise_pred = noise_pred.sum(1)
            predicted_variance = predicted_variance.mean(1)
        elif reduction == 'alternate':
            noise_pred = noise_pred[:,i%num_prompts]
            predicted_variance = predicted_variance[:,i%num_prompts]
        else:
            raise ValueError('Reduction must be either `mean` or `alternate`')
        noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

        # compute the previous noisy sample x_t -> x_t-1
        noisy_images = model.scheduler.step(
            noise_pred, t, noisy_images, generator=generator, return_dict=False
        )[0]

    # Return denoised images
    return noisy_images







@torch.no_grad()
def sample_stage_2(model,
                   image,
                   prompt_embeds,
                   negative_prompt_embeds, 
                   views,
                   ref_im=None,
                   num_inference_steps=100,
                   guidance_scale=7.0,
                   reduction='mean',
                   noise_level=50,
                   generator=None):

    # Params
    batch_size = 1      # TODO: Support larger batch sizes, maybe
    num_prompts = prompt_embeds.shape[0]
    height = model.unet.config.sample_size
    width = model.unet.config.sample_size
    device = model.device
    num_images_per_prompt = 1

    # For CFG
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # Get timesteps
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = model.scheduler.timesteps

    num_channels = model.unet.config.in_channels // 2
    noisy_images = model.prepare_intermediate_images(
        batch_size * num_images_per_prompt,
        num_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
    )

    # Resize ref image to correct size
    if ref_im is not None:
        ref_im = TF.resize(ref_im, height)
        ref_im = ref_im.to(noisy_images.device).to(noisy_images.dtype)

    # Prepare upscaled image and noise level
    image = model.preprocess_image(image, num_images_per_prompt, device)
    upscaled = F.interpolate(image, (height, width), mode="bilinear", align_corners=True)

    noise_level = torch.tensor([noise_level] * upscaled.shape[0], device=upscaled.device)
    noise = randn_tensor(upscaled.shape, generator=generator, device=upscaled.device, dtype=upscaled.dtype)
    upscaled = model.image_noising_scheduler.add_noise(upscaled, noise, timesteps=noise_level)

    # Condition on noise level, for each model input
    noise_level = torch.cat([noise_level] * num_prompts * 2)

    # Denoising Loop
    for i, t in enumerate(tqdm(timesteps)):
        # If solving an inverse problem, then project x_t so
        # that first component matches reference image's first component
        if ref_im is not None:
            # Inject noise to reference image
            alpha_cumprod = model.scheduler.alphas_cumprod[t]
            ref_noisy = torch.sqrt(alpha_cumprod) * ref_im + \
                        torch.sqrt(1 - alpha_cumprod) * torch.randn_like(ref_im)

            # Replace 1st component of x_t with 1st component of noisy ref image
            ref_noisy_component = views[0].inverse_view(ref_noisy)
            noisy_images_component = views[1].inverse_view(noisy_images[0])
            noisy_images = ref_noisy_component + noisy_images_component

            # Add back batch dim
            noisy_images = noisy_images[None]

        # Cat noisy image with upscaled conditioning image
        model_input = torch.cat([noisy_images, upscaled], dim=1)

        # Apply views to noisy_image
        viewed_inputs = []
        for view_fn in views:
            viewed_inputs.append(view_fn.view(model_input[0]))
        viewed_inputs = torch.stack(viewed_inputs)

        # Duplicate inputs for CFG
        # Model input is: [ neg_0, neg_1, ..., pos_0, pos_1, ... ]
        model_input = torch.cat([viewed_inputs] * 2)
        model_input = model.scheduler.scale_model_input(model_input, t)

        # predict the noise residual
        noise_pred = model.unet(
            model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            class_labels=noise_level,
            cross_attention_kwargs=None,
            return_dict=False,
        )[0]

        # Extract uncond (neg) and cond noise estimates
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        # Invert the unconditional (negative) estimates
        # TODO: pretty sure you can combine these into one loop
        inverted_preds = []
        for pred, view in zip(noise_pred_uncond, views):
            inverted_pred = view.inverse_view(pred)
            inverted_preds.append(inverted_pred)
        noise_pred_uncond = torch.stack(inverted_preds)

        # Invert the conditional estimates
        inverted_preds = []
        for pred, view in zip(noise_pred_text, views):
            inverted_pred = view.inverse_view(pred)
            inverted_preds.append(inverted_pred)
        noise_pred_text = torch.stack(inverted_preds)

        # Split predicted noise and predicted variances
        noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1] // 2, dim=1)
        noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1] // 2, dim=1)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Combine noise estimates (and variance estimates)
        noise_pred = noise_pred.view(-1,num_prompts,3,256,256)
        predicted_variance = predicted_variance.view(-1,num_prompts,3,256,256)
        if reduction == 'mean':
            noise_pred = noise_pred.mean(1)
            predicted_variance = predicted_variance.mean(1)
        elif reduction == 'sum':
            # For factorized diffusion
            noise_pred = noise_pred.sum(1)
            predicted_variance = predicted_variance.mean(1)
        elif reduction == 'alternate':
            noise_pred = noise_pred[:,i%num_prompts]
            predicted_variance = predicted_variance[:,i%num_prompts]

        noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

        # compute the previous noisy sample x_t -> x_t-1
        noisy_images = model.scheduler.step(
            noise_pred, t, noisy_images, generator=generator, return_dict=False
        )[0]

    # Return denoised images
    return noisy_images

@torch.no_grad()
def sample_sdxl_single_stage(model,
                             prompt_embeds,
                             negative_prompt_embeds,
                             pooled_prompt_embeds,
                             negative_pooled_prompt_embeds,
                             views,
                             height=1024,
                             width=1024,
                             ref_im=None,
                             num_inference_steps=50,
                             guidance_scale=7.5,
                             reduction='mean',
                             generator=None):
    """
    Sample from SDXL with multi-view constraints.
    """
    from tqdm import tqdm
    
    # Params
    batch_size = 1
    num_prompts = len(views)
    device = model.device if hasattr(model, 'device') else torch.device('cuda')
    dtype = torch.float16
    
    print(f"Sampling with {num_prompts} views, guidance scale {guidance_scale}")
    
    # Setup timesteps
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = model.scheduler.timesteps
    
    # Prepare latents
    num_channels_latents = model.unet.config.in_channels
    latents = model.prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
    )
    
    print(f"Initial latents shape: {latents.shape}")
    
    # Prepare added time ids & embeddings
    original_size = (height, width)
    target_size = (height, width)
    crops_coords_top_left = (0, 0)
    
    # Create add_time_ids for each view (both negative and positive)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
    add_time_ids = add_time_ids.repeat(num_prompts * 2, 1)  # *2 for CFG
    
    # Resize ref image if needed
    if ref_im is not None:
        ref_im = TF.resize(ref_im, height // 8)  # SDXL uses 8x downsampling
        ref_im = ref_im.to(device).to(dtype)
    
    # Denoising Loop
    for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
        # If solving inverse problem, project latents
        if ref_im is not None:
            ref_latent = model.vae.encode(ref_im[None]).latent_dist.sample()
            ref_latent = ref_latent * model.vae.config.scaling_factor
            
            alpha_prod_t = model.scheduler.alphas_cumprod[t]
            ref_noisy = torch.sqrt(alpha_prod_t) * ref_latent + \
                       torch.sqrt(1 - alpha_prod_t) * torch.randn_like(ref_latent)
            
            ref_component = views[0].inverse_view(ref_noisy[0])
            latents_component = views[1].inverse_view(latents[0])
            latents = (ref_component + latents_component)[None]
        
        # Apply views to latents
        viewed_latents = []
        for view_fn in views:
            viewed_latents.append(view_fn.view(latents[0]))
        viewed_latents = torch.stack(viewed_latents)  # [num_prompts, C, H, W]
        
        # Duplicate for CFG: [viewed_0, viewed_1, ..., viewed_0, viewed_1, ...]
        model_input = torch.cat([viewed_latents, viewed_latents])  # [2*num_prompts, C, H, W]
        model_input = model.scheduler.scale_model_input(model_input, t)
        
        # Prepare conditioning
        added_cond_kwargs = {
            "text_embeds": torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds]),
            "time_ids": add_time_ids
        }
        
        # Predict noise
        noise_pred = model.unet(
            model_input,
            t,
            encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds]),
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        
        # Split into uncond and cond
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        
        # Invert unconditional estimates
        inverted_uncond = []
        for pred, view in zip(noise_pred_uncond, views):
            inverted_uncond.append(view.inverse_view(pred))
        noise_pred_uncond = torch.stack(inverted_uncond)
        
        # Invert conditional estimates  
        inverted_cond = []
        for pred, view in zip(noise_pred_text, views):
            inverted_cond.append(view.inverse_view(pred))
        noise_pred_text = torch.stack(inverted_cond)
        
        # Apply CFG
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Reduce predictions across views
        if reduction == 'mean':
            noise_pred = noise_pred.mean(0, keepdim=True)
        elif reduction == 'sum':
            noise_pred = noise_pred.sum(0, keepdim=True)
        elif reduction == 'alternate':
            noise_pred = noise_pred[i % num_prompts:i % num_prompts + 1]
        else:
            raise ValueError('Reduction must be `mean`, `sum`, or `alternate`')
        
        # Denoise step
        latents = model.scheduler.step(
            noise_pred, t, latents, generator=generator, return_dict=False
        )[0]
    
    print(f"Final latents shape: {latents.shape}, range: [{latents.min():.3f}, {latents.max():.3f}]")
    
    # Decode latents with proper scaling and dtype handling
    latents = latents / model.vae.config.scaling_factor
    
    # SDXL VAE needs float32 for numerical stability to avoid NaN issues
    needs_upcasting = model.vae.dtype == torch.float16 and model.vae.config.force_upcast
    
    if needs_upcasting:
        model.vae.to(dtype=torch.float32)
        latents = latents.float()
    
    image = model.vae.decode(latents, return_dict=False)[0]
    
    if needs_upcasting:
        model.vae.to(dtype=torch.float16)
    
    # Check for NaN and clamp
    if torch.isnan(image).any():
        print("WARNING: NaN detected in decoded image, replacing with zeros")
        image = torch.nan_to_num(image, nan=0.0)
    
    print(f"Decoded image shape: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]")
    
    # SDXL VAE outputs are typically in range [-1, 1] after decoding
    # Clamp to ensure valid range
    image = torch.clamp(image, -1.0, 1.0)
    
    return image


# variant 2

import clip
import torch.nn.functional as F

# Replace the sample_sdxl_adaptive_progressive function in samplers.py with this fixed version:

@torch.no_grad()
def sample_sdxl_adaptive_progressive(
    model,
    prompt_embeds,
    negative_prompt_embeds,
    pooled_prompt_embeds,
    negative_pooled_prompt_embeds,
    views,
    prompts,  # Need text prompts for CLIP scoring
    height=1024,
    width=1024,
    ref_im=None,
    num_inference_steps=50,
    guidance_scale=7.5,
    reduction='adaptive',
    generator=None,
    clip_model=None,
    clip_preprocess=None
):
    """
    Adaptive progressive sampling with three stages:
    1. Structure establishment (equal weights, high guidance)
    2. View balancing (adaptive weights from CLIP scores)
    3. Quality refinement (transition to identity-dominant)
    """
    from tqdm import tqdm
    import torchvision.transforms.functional as TF
    
    # Load CLIP for scoring if not provided
    if clip_model is None:
        import clip
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=model.device)
    
    # Params
    batch_size = 1
    num_prompts = len(views)
    device = model.device if hasattr(model, 'device') else torch.device('cuda')
    dtype = prompt_embeds.dtype  # Use the dtype from embeddings
    
    print(f"Adaptive progressive sampling with {num_prompts} views")
    print(f"  Stage 1: Timesteps {int(num_inference_steps*0.7)}-{num_inference_steps} (Structure)")
    print(f"  Stage 2: Timesteps {int(num_inference_steps*0.3)}-{int(num_inference_steps*0.7)} (Balancing)")
    print(f"  Stage 3: Timesteps 0-{int(num_inference_steps*0.3)} (Refinement)")
    
    # Setup timesteps
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = model.scheduler.timesteps
    
    # Define stage boundaries
    stage1_start = int(num_inference_steps * 0.7)
    stage2_start = int(num_inference_steps * 0.3)
    
    # Prepare latents
    num_channels_latents = model.unet.config.in_channels
    latents = model.prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
    )
    
    print(f"Initial latents shape: {latents.shape}, dtype: {latents.dtype}")
    
    # Prepare added time ids
    original_size = (height, width)
    target_size = (height, width)
    crops_coords_top_left = (0, 0)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
    add_time_ids = add_time_ids.repeat(num_prompts * 2, 1)
    
    # Initialize view weights
    view_weights = torch.ones(num_prompts, device=device, dtype=torch.float32) / num_prompts
    
    # Denoising Loop
    for step_idx, t in enumerate(tqdm(timesteps, desc="Denoising")):
        # Determine current stage and adjust strategy
        if step_idx < stage2_start:
            stage = 3  # Refinement
            current_guidance = guidance_scale * 0.7
        elif step_idx < stage1_start:
            stage = 2  # Balancing
            current_guidance = guidance_scale
        else:
            stage = 1  # Structure
            current_guidance = guidance_scale * 1.2
        
        # Apply views to latents - ensure dtype is preserved
        viewed_latents = []
        for view_fn in views:
            viewed = view_fn.view(latents[0])
            # Ensure viewed tensor has correct dtype
            if viewed.dtype != dtype:
                viewed = viewed.to(dtype)
            viewed_latents.append(viewed)
        viewed_latents = torch.stack(viewed_latents)
        
        # Duplicate for CFG
        model_input = torch.cat([viewed_latents, viewed_latents])
        model_input = model.scheduler.scale_model_input(model_input, t)
        
        # Ensure model_input has correct dtype
        if model_input.dtype != dtype:
            model_input = model_input.to(dtype)
        
        # Prepare conditioning
        added_cond_kwargs = {
            "text_embeds": torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds]),
            "time_ids": add_time_ids
        }
        
        # Predict noise
        noise_pred = model.unet(
            model_input,
            t,
            encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds]),
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        
        # Split into uncond and cond
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        
        # Invert unconditional estimates
        inverted_uncond = []
        for pred, view in zip(noise_pred_uncond, views):
            inverted = view.inverse_view(pred)
            if inverted.dtype != dtype:
                inverted = inverted.to(dtype)
            inverted_uncond.append(inverted)
        noise_pred_uncond = torch.stack(inverted_uncond)
        
        # Invert conditional estimates
        inverted_cond = []
        for pred, view in zip(noise_pred_text, views):
            inverted = view.inverse_view(pred)
            if inverted.dtype != dtype:
                inverted = inverted.to(dtype)
            inverted_cond.append(inverted)
        noise_pred_text = torch.stack(inverted_cond)
        
        # Apply CFG
        noise_pred = noise_pred_uncond + current_guidance * (noise_pred_text - noise_pred_uncond)
        
        # === ADAPTIVE VIEW WEIGHTING ===
        if stage == 2:  # Balancing stage
            # Every N steps, compute CLIP scores and adjust weights
            if step_idx % 5 == 0:
                # Decode current latents to compute CLIP scores
                with torch.no_grad():
                    # Create a copy of latents for CLIP scoring to avoid modifying original
                    latents_for_clip = latents.clone()
                    
                    # Scale latents for VAE
                    scaled_latents = latents_for_clip / model.vae.config.scaling_factor
                    
                    # Handle VAE dtype - save original state
                    original_vae_dtype = model.vae.dtype
                    needs_upcasting = original_vae_dtype == torch.float16 and model.vae.config.force_upcast
                    
                    if needs_upcasting:
                        model.vae.to(dtype=torch.float32)
                        scaled_latents = scaled_latents.float()
                    
                    current_image = model.vae.decode(scaled_latents, return_dict=False)[0]
                    
                    # Restore VAE dtype
                    if needs_upcasting:
                        model.vae.to(dtype=original_vae_dtype)
                    
                    current_image = torch.clamp(current_image, -1.0, 1.0)
                    
                    # Compute CLIP scores for each view
                    clip_scores = []
                    for view_fn, prompt in zip(views, prompts):
                        # Apply view transformation
                        viewed_img = view_fn.view(current_image[0])
                        
                        # Normalize for CLIP [0, 1]
                        viewed_img_normalized = (viewed_img + 1.0) / 2.0
                        viewed_img_pil = TF.to_pil_image(viewed_img_normalized.cpu().clamp(0, 1))
                        
                        # Get CLIP image features
                        clip_input = clip_preprocess(viewed_img_pil).unsqueeze(0).to(device)
                        image_features = clip_model.encode_image(clip_input)
                        
                        # Get CLIP text features
                        import clip
                        text_input = clip.tokenize([prompt]).to(device)
                        text_features = clip_model.encode_text(text_input)
                        
                        # Compute similarity
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        similarity = (image_features @ text_features.T).item()
                        clip_scores.append(similarity)
                    
                    clip_scores = torch.tensor(clip_scores, device=device, dtype=torch.float32)
                    
                    # Inverse weighting: boost views with lower scores
                    # Normalize scores to [0, 1] range
                    clip_scores_norm = (clip_scores - clip_scores.min()) / (clip_scores.max() - clip_scores.min() + 1e-8)
                    
                    # Inverse weighting with temperature
                    temperature = 2.0
                    inverse_scores = 1.0 - clip_scores_norm
                    view_weights = F.softmax(inverse_scores / temperature, dim=0)
                    
                    print(f"\n  Step {step_idx}: CLIP scores: {clip_scores.cpu().numpy()}")
                    print(f"  Adaptive weights: {view_weights.cpu().numpy()}")
        
        elif stage == 3:  # Refinement stage
            # Gradually transition to identity-dominant
            progress = (step_idx / stage2_start)  # 0 to 1 through stage 3
            identity_weight = 0.5 + 0.5 * progress  # 0.5 to 1.0
            other_weight = (1.0 - identity_weight) / (num_prompts - 1)
            
            view_weights = torch.tensor(
                [identity_weight] + [other_weight] * (num_prompts - 1),
                device=device,
                dtype=torch.float32
            )
        
        else:  # Structure stage
            # Equal weights
            view_weights = torch.ones(num_prompts, device=device, dtype=torch.float32) / num_prompts
        
        # Apply weighted reduction - convert weights to correct dtype
        weights_expanded = view_weights.view(-1, 1, 1, 1).to(dtype)
        noise_pred = (noise_pred * weights_expanded).sum(0, keepdim=True)
        
        # Ensure noise_pred has correct dtype before scheduler step
        if noise_pred.dtype != dtype:
            noise_pred = noise_pred.to(dtype)
        
        # Denoise step
        latents = model.scheduler.step(
            noise_pred, t, latents, generator=generator, return_dict=False
        )[0]
        
        # Ensure latents maintain correct dtype
        if latents.dtype != dtype:
            latents = latents.to(dtype)
    
    print(f"\nFinal latents shape: {latents.shape}, range: [{latents.min():.3f}, {latents.max():.3f}]")
    
    # Decode final latents
    latents = latents / model.vae.config.scaling_factor
    
    needs_upcasting = model.vae.dtype == torch.float16 and model.vae.config.force_upcast
    if needs_upcasting:
        model.vae.to(dtype=torch.float32)
        latents = latents.float()
    
    image = model.vae.decode(latents, return_dict=False)[0]
    
    if needs_upcasting:
        model.vae.to(dtype=torch.float16)
    
    if torch.isnan(image).any():
        print("WARNING: NaN detected in decoded image")
        image = torch.nan_to_num(image, nan=0.0)
    
    image = torch.clamp(image, -1.0, 1.0)
    
    return image


# variant 3

# Add to samplers.py

import numpy as np
from scipy.linalg import polar
# Replace the sample_sdxl_frequency_orthogonal function in samplers.py with this fixed version:

@torch.no_grad()
def sample_sdxl_frequency_orthogonal(
    model,
    prompt_embeds,
    negative_prompt_embeds,
    pooled_prompt_embeds,
    negative_pooled_prompt_embeds,
    views,
    height=1024,
    width=1024,
    ref_im=None,
    num_inference_steps=50,
    guidance_scale=7.5,
    generator=None,
    momentum_beta=0.9,
    orthogonal_projection=True,
    frequency_scheduling=True,
):
    """
    Frequency-decomposed sampling with explicit orthogonal projection and momentum.
    """
    from tqdm import tqdm
    
    try:
        # Params
        batch_size = 1
        num_prompts = len(views)
        device = model.device if hasattr(model, 'device') else torch.device('cuda')
        dtype = prompt_embeds.dtype
        
        print(f"Frequency-Orthogonal sampling with {num_prompts} views")
        print(f"  Orthogonal projection: {orthogonal_projection}")
        print(f"  Frequency scheduling: {frequency_scheduling}")
        print(f"  Momentum beta: {momentum_beta}")
        
        # Setup timesteps
        model.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = model.scheduler.timesteps
        
        # Prepare latents
        num_channels_latents = model.unet.config.in_channels
        latents = model.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
        )
        
        print(f"Initial latents shape: {latents.shape}, dtype: {latents.dtype}")
        
        # Prepare added time ids
        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
        add_time_ids = add_time_ids.repeat(num_prompts * 2, 1)
        
        # Initialize momentum buffer
        momentum_buffer = None
        
        def get_frequency_weights(step_idx, num_steps, num_views):
            """Compute frequency-aware weights for views."""
            progress = step_idx / num_steps
            
            if not frequency_scheduling:
                return torch.ones(num_views, device=device, dtype=torch.float32) / num_views
            
            # Identity view weight increases with progress
            identity_weight = 0.4 + 0.3 * progress
            other_weight = (1.0 - identity_weight) / (num_views - 1) if num_views > 1 else 0.0
            
            weights = torch.tensor(
                [identity_weight] + [other_weight] * (num_views - 1),
                device=device,
                dtype=torch.float32
            )
            
            return weights
        
        # Replace the project_to_orthogonal_manifold function in samplers.py:

        def project_to_orthogonal_manifold(noise_estimates):
            """
            FIXED: The paper requires view TRANSFORMATIONS to be orthogonal,
            not the noise estimates themselves. 
            
            Views like flip, rotate are already orthogonal transformations by construction.
            Normalizing the noise estimates destroys their necessary magnitude.
            
            This function now does NOTHING when orthogonal_projection=True,
            or can be safely removed. Keeping it for API compatibility.
            """
            if not orthogonal_projection:
                return noise_estimates
            
            # DO NOTHING - views are already orthogonal by construction
            # The paper's constraint is that v(x) must be an orthogonal transformation,
            # not that we need to project the combined noise estimates
            return noise_estimates
        
        # Denoising Loop
        for step_idx, t in enumerate(tqdm(timesteps, desc="Denoising")):
            # Get frequency-aware weights
            freq_weights = get_frequency_weights(step_idx, num_inference_steps, num_prompts)
            
            # Apply views to latents
            viewed_latents = []
            for view_fn in views:
                viewed = view_fn.view(latents[0])
                if viewed.dtype != dtype:
                    viewed = viewed.to(dtype)
                viewed_latents.append(viewed)
            viewed_latents = torch.stack(viewed_latents)
            
            # Duplicate for CFG
            model_input = torch.cat([viewed_latents, viewed_latents])
            model_input = model.scheduler.scale_model_input(model_input, t)
            
            if model_input.dtype != dtype:
                model_input = model_input.to(dtype)
            
            # Prepare conditioning
            added_cond_kwargs = {
                "text_embeds": torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds]),
                "time_ids": add_time_ids
            }
            
            # Predict noise
            noise_pred = model.unet(
                model_input,
                t,
                encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds]),
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            
            # Split into uncond and cond
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            
            # Invert unconditional estimates
            inverted_uncond = []
            for pred, view in zip(noise_pred_uncond, views):
                inverted = view.inverse_view(pred)
                if inverted.dtype != dtype:
                    inverted = inverted.to(dtype)
                inverted_uncond.append(inverted)
            noise_pred_uncond = torch.stack(inverted_uncond)
            
            # Invert conditional estimates
            inverted_cond = []
            for pred, view in zip(noise_pred_text, views):
                inverted = view.inverse_view(pred)
                if inverted.dtype != dtype:
                    inverted = inverted.to(dtype)
                inverted_cond.append(inverted)
            noise_pred_text = torch.stack(inverted_cond)
            
            # Apply CFG
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Frequency-aware weighted combination
            weights_expanded = freq_weights.view(-1, 1, 1, 1).to(dtype)
            noise_pred_combined = (noise_pred * weights_expanded).sum(0, keepdim=True)
            
            # Orthogonal projection
            noise_pred_combined = project_to_orthogonal_manifold(noise_pred_combined)
            
            # Momentum stabilization
            if momentum_buffer is None:
                momentum_buffer = noise_pred_combined.clone()
            else:
                momentum_buffer = momentum_beta * momentum_buffer + (1 - momentum_beta) * noise_pred_combined
                noise_pred_combined = momentum_buffer
            
            # Log every 10 steps
            if step_idx % 10 == 0:
                print(f"  Step {step_idx}/{num_inference_steps}: weights={freq_weights.cpu().numpy()}, "
                      f"noise_range=[{noise_pred_combined.min():.3f}, {noise_pred_combined.max():.3f}]")
            
            # Ensure correct dtype
            if noise_pred_combined.dtype != dtype:
                noise_pred_combined = noise_pred_combined.to(dtype)
            
            # Denoise step
            latents = model.scheduler.step(
                noise_pred_combined, t, latents, generator=generator, return_dict=False
            )[0]
            
            if latents.dtype != dtype:
                latents = latents.to(dtype)
        
        print(f"\nCompleted denoising loop")
        print(f"Final latents shape: {latents.shape}, range: [{latents.min():.3f}, {latents.max():.3f}]")
        
        # Decode final latents
        print(f"Decoding latents...")
        latents = latents / model.vae.config.scaling_factor
        
        needs_upcasting = model.vae.dtype == torch.float16 and model.vae.config.force_upcast
        if needs_upcasting:
            print(f"  Upcasting VAE to float32 for decoding...")
            model.vae.to(dtype=torch.float32)
            latents = latents.float()
        
        image = model.vae.decode(latents, return_dict=False)[0]
        
        if needs_upcasting:
            print(f"  Restoring VAE to float16...")
            model.vae.to(dtype=torch.float16)
        
        print(f"Decoded image shape: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]")
        
        # Check for NaN
        if torch.isnan(image).any():
            print("WARNING: NaN detected in decoded image, replacing with zeros")
            image = torch.nan_to_num(image, nan=0.0)
        
        # Clamp to valid range
        image = torch.clamp(image, -1.0, 1.0)
        
        print(f"Final image shape: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"Returning image...\n")
        
        return image
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR in sample_sdxl_frequency_orthogonal:")
        print(f"{'='*60}")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        raise