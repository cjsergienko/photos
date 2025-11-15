#!/usr/bin/env python3
"""
Pre-download AI models for photo restoration.
Run this before starting the web app to avoid timeouts.
"""

import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer

print("=" * 60)
print("Downloading AI Models for Photo Restoration")
print("=" * 60)
print("\nThis will download approximately 500MB of models.")
print("It may take 5-10 minutes depending on your internet speed.\n")

# Determine device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")

# Download Real-ESRGAN model
print("1/4 Downloading Real-ESRGAN model (~64MB)...")
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False if device == 'cpu' else True,
    device=device
)
print("✓ Real-ESRGAN model downloaded\n")

# Download GFPGAN models
print("2/4 Downloading GFPGAN face detection model (~104MB)...")
print("3/4 Downloading GFPGAN parsing model (~81MB)...")
print("4/4 Downloading GFPGAN restoration model (~332MB)...")
face_enhancer = GFPGANer(
    model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
    upscale=4,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=upsampler,
    device=device
)
print("✓ All GFPGAN models downloaded\n")

print("=" * 60)
print("SUCCESS! All models downloaded successfully.")
print("=" * 60)
print("\nYou can now start the web app with:")
print("  python app.py")
print("\nThe first photo restoration will be fast!")
