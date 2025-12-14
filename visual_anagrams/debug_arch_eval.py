#!/usr/bin/env python
"""
Debug script for architecture comparison evaluation.

This helps diagnose issues with evaluate_arch_comparison.py
"""

import sys
from pathlib import Path
import pickle
import torch

def check_results_directory(results_dir):
    """Check if results directory has the expected structure."""
    results_dir = Path(results_dir)
    
    print("="*60)
    print("DEBUGGING ARCHITECTURE COMPARISON")
    print("="*60)
    print(f"\nChecking: {results_dir}")
    print(f"Exists: {results_dir.exists()}")
    
    if not results_dir.exists():
        print("❌ Directory does not exist!")
        return False
    
    # Check for metadata file (single file at root)
    print("\n" + "="*60)
    print("LOOKING FOR METADATA FILE")
    print("="*60)
    
    metadata_path = results_dir / 'metadata.pkl'
    print(f"Looking for: {metadata_path}")
    print(f"Exists: {metadata_path.exists()}")
    
    if not metadata_path.exists():
        print("❌ No metadata.pkl file found at root!")
        print("\nExpected structure:")
        print("  results/flip.dog_cat.variant0/")
        print("    ├── metadata.pkl")
        print("    ├── 0/")
        print("    │   ├── sample_64.png")
        print("    │   ├── sample_256.png")
        print("    │   └── sample_1024.png")
        print("    ├── 1/")
        print("    └── ...")
        return False
    
    # Load and check metadata
    print(f"\nChecking metadata file: {metadata_path}")
    
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        print("✓ Successfully loaded metadata")
        print(f"  Keys: {list(metadata.keys())}")
        
        # Check views
        if 'views' in metadata:
            views = metadata['views']
            print(f"  Views: {len(views)} found")
            for i, view in enumerate(views):
                print(f"    [{i}] {view.__class__.__name__}")
        else:
            print("  ❌ No 'views' key in metadata!")
            return False
        
        # Check args
        if 'args' in metadata:
            args = metadata['args']
            print(f"  Args: Found")
            if hasattr(args, 'prompts'):
                print(f"    prompts: {args.prompts}")
            if hasattr(args, 'style'):
                print(f"    style: {args.style}")
            
            # Reconstruct full prompts
            prompts = [f'{args.style} {p}'.strip() for p in args.prompts]
            print(f"  Full prompts: {prompts}")
        else:
            print("  ❌ No 'args' key in metadata!")
            return False
            
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check for sample directories
    print("\n" + "="*60)
    print("CHECKING FOR SAMPLE DIRECTORIES")
    print("="*60)
    
    sample_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    print(f"Found {len(sample_dirs)} numbered sample directories")
    
    if len(sample_dirs) == 0:
        print("❌ No sample directories found!")
        print("Expected directories like: 0, 1, 2, ...")
        return False
    
    # Check first sample directory
    first_sample = sample_dirs[0]
    print(f"\nChecking first sample: {first_sample}")
    
    for image_size in [64, 256, 1024]:
        image_path = first_sample / f'sample_{image_size}.png'
        exists = image_path.exists()
        print(f"  sample_{image_size}.png: {'✓' if exists else '❌'}")
    
    # Try to load CLIP models
    print("\n" + "="*60)
    print("TESTING CLIP MODEL LOADING")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    try:
        import clip
        
        print("\nLoading RN50 (CNN)...")
        cnn_model, cnn_preprocess = clip.load("RN50", device=device)
        print("✓ RN50 loaded successfully")
        
        print("\nLoading ViT-B/32 (Transformer)...")
        vit_model, vit_preprocess = clip.load("ViT-B/32", device=device)
        print("✓ ViT-B/32 loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading CLIP: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Try to actually evaluate one sample
    print("\n" + "="*60)
    print("TESTING SAMPLE EVALUATION")
    print("="*60)
    
    try:
        from PIL import Image
        import torchvision.transforms.functional as TF
        
        # Load image
        image_path = first_sample / 'sample_1024.png'
        if not image_path.exists():
            print(f"❌ {image_path} not found")
            return False
        
        print(f"Loading: {image_path}")
        image_pil = Image.open(image_path).convert('RGB')
        print(f"✓ Image loaded: {image_pil.size}")
        
        # Convert to tensor
        image_tensor = TF.to_tensor(image_pil) * 2 - 1
        image_tensor = image_tensor.to(device)
        print(f"✓ Converted to tensor: {image_tensor.shape}")
        
        # Try to apply first view
        view = views[0]
        print(f"\nApplying view: {view.__class__.__name__}")
        
        viewed_tensor = view.view(image_tensor)
        print(f"✓ View applied: {viewed_tensor.shape}")
        
        # Try to get CLIP embeddings
        viewed_normalized = (viewed_tensor + 1) / 2
        viewed_normalized = torch.clamp(viewed_normalized, 0, 1)
        viewed_pil = TF.to_pil_image(viewed_normalized.cpu())
        
        image_input = cnn_preprocess(viewed_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = cnn_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        print(f"✓ CLIP encoding successful: {image_features.shape}")
        
        print("\n" + "="*60)
        print("✓ ALL CHECKS PASSED!")
        print("="*60)
        print("\nYou should be able to run:")
        print(f"  python -m visual_anagrams.evaluate_arch_comparison --results_dir {results_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during evaluation test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python debug_arch_eval.py <results_dir>")
        print("Example: python debug_arch_eval.py results/flip.dog_cat.variant0")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    success = check_results_directory(results_dir)
    
    if not success:
        print("\n❌ Issues found. Please fix them before running evaluate_arch_comparison.py")
        sys.exit(1)
    else:
        sys.exit(0)