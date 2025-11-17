#!/bin/bash

# Photo Restoration App - Installation Script for macOS
# This script sets up the complete environment for the photo restoration application

set -e  # Exit on any error

echo "============================================"
echo "Photo Restoration App - macOS Installation"
echo "============================================"
echo ""

# Check if we're on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This script is designed for macOS only."
    exit 1
fi

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    echo "Please install Python 3 using Homebrew: brew install python3"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $PYTHON_VERSION"

# Check for pip
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed."
    echo "Please install pip: python3 -m ensurepip --upgrade"
    exit 1
fi
echo "✓ Found pip3"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip
echo "✓ pip upgraded"

# Install core dependencies
echo ""
echo "Installing core dependencies..."
pip install flask opencv-python-headless numpy werkzeug
echo "✓ Core dependencies installed"

# Install AI restoration dependencies (optional but recommended)
echo ""
echo "Installing AI restoration dependencies..."
echo "This may take a while and download large model files..."

# Install PyTorch (CPU version for macOS)
pip install torch torchvision torchaudio

# Install Real-ESRGAN and GFPGAN
pip install basicsr realesrgan gfpgan

# Install DeOldify dependencies
pip install deoldify fastai==1.0.61

echo "✓ AI restoration dependencies installed"

# Create necessary directories
echo ""
echo "Creating required directories..."
mkdir -p uploads results
echo "✓ Directories created"

# Pre-download AI models
echo ""
echo "Pre-downloading AI models (this may take a few minutes)..."
echo "This ensures faster first-time usage of the app."

python3 << 'EOF'
import os
import urllib.request
import sys

def download_file(url, destination):
    """Download file with progress indicator"""
    print(f"  Downloading: {os.path.basename(destination)}")

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r    Progress: {percent}%")
        sys.stdout.flush()

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    urllib.request.urlretrieve(url, destination, progress_hook)
    print(" - Done!")

# Create weights directory
weights_dir = os.path.expanduser("~/.cache/gfpgan/weights")
os.makedirs(weights_dir, exist_ok=True)

# Download GFPGAN model
gfpgan_path = os.path.join(weights_dir, "GFPGANv1.3.pth")
if not os.path.exists(gfpgan_path):
    download_file(
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        gfpgan_path
    )
else:
    print(f"  ✓ GFPGAN model already downloaded")

# Download Real-ESRGAN model
realesrgan_dir = os.path.expanduser("~/.cache/realesrgan")
os.makedirs(realesrgan_dir, exist_ok=True)
realesrgan_path = os.path.join(realesrgan_dir, "RealESRGAN_x4plus.pth")
if not os.path.exists(realesrgan_path):
    download_file(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        realesrgan_path
    )
else:
    print(f"  ✓ Real-ESRGAN model already downloaded")

# Download face detection model (required by GFPGAN)
detection_path = os.path.join(weights_dir, "detection_Resnet50_Final.pth")
if not os.path.exists(detection_path):
    download_file(
        "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        detection_path
    )
else:
    print(f"  ✓ Face detection model already downloaded")

# Download parsing model (required by GFPGAN)
parsing_path = os.path.join(weights_dir, "parsing_parsenet.pth")
if not os.path.exists(parsing_path):
    download_file(
        "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
        parsing_path
    )
else:
    print(f"  ✓ Face parsing model already downloaded")

print("\n✓ All AI models pre-downloaded successfully!")
EOF

# Set permissions
chmod +x install.sh 2>/dev/null || true

echo ""
echo "============================================"
echo "Installation Complete!"
echo "============================================"
echo ""
echo "To start the application:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run the server:"
echo "     python app.py"
echo ""
echo "  3. Open your browser and go to:"
echo "     http://localhost:8080"
echo ""
echo "Features:"
echo "  - Batch Enhancement: Fast OpenCV-based photo restoration"
echo "  - Fine Tuning: Advanced AI restoration with:"
echo "    * DeOldify colorization"
echo "    * GFPGAN face restoration"
echo "    * Real-ESRGAN image enhancement"
echo ""
echo "All AI models are pre-downloaded and ready to use!"
echo "============================================"
