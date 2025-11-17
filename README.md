# Photo Restoration Web App

Web application for restoring and enhancing old photos using AI neural networks combined with real-time adjustments.

## Features

### Batch Enhancement
- **Fast OpenCV Processing**: Quick enhancement for multiple photos
- **Real-Time Adjustments**: Live slider controls for instant visual feedback
- **Batch Upload**: Process multiple photos at once
- **Customizable Results**: 5 adjustment sliders (Contrast, Noise Reduction, Sharpen, Saturation, Brightness)
- **Download All**: Batch download as ZIP file

### Fine Tuning (Advanced AI)
- **DeOldify Colorization**: AI-powered colorization of black & white photos
- **GFPGAN Face Restoration**: Deep learning enhancement for facial features
- **Real-ESRGAN Enhancement**: State-of-the-art image quality improvement
- **Artifact Removal**: Intelligent dust spot and scratch detection
- **Heavy Processing**: Designed for single photos requiring maximum quality

## How It Works

**Two-Step Restoration Process:**

### Step 1: AI Enhancement (Server-Side)
- **Real-ESRGAN Neural Network**: State-of-the-art super-resolution AI
- Runs once when you upload your photo
- Downloads AI model automatically on first run (~11MB)
- Upscales and enhances image quality using deep learning
- Processing time: First run ~30-60 seconds (model download), then ~5-10 seconds per photo

### Step 2: Live Adjustments (Client-Side)
- **Instant Browser-Based Processing**: No server round-trips needed
- Adjust sliders and see results immediately (Canvas API)
- **5 Control Sliders:**
  - **Contrast**: Adjust image contrast
  - **Noise Reduction**: Reduce grain and artifacts
  - **Sharpness**: Enhance fine details
  - **Color Saturation**: Boost or reduce color vibrancy
  - **Brightness**: Fine-tune overall brightness
- All adjustments happen in real-time in your browser

## Quick Start (macOS)

### Automated Installation

1. **Clone this repository:**
```bash
git clone https://github.com/cjsergienko/photos.git
cd photos
```

2. **Run the install script:**
```bash
./install.sh
```

This will automatically:
- Create a virtual environment
- Install all dependencies (Flask, OpenCV, PyTorch, GFPGAN, Real-ESRGAN, DeOldify)
- Create necessary directories
- Set up the complete AI restoration pipeline

3. **Start the application:**
```bash
source venv/bin/activate
python app.py
```

4. **Open your browser:**
   - Navigate to http://localhost:8080
   - Use sidebar to switch between Batch Enhancement and Fine Tuning

### Using Claude Code (Alternative)

1. **Clone this repository:**
```bash
git clone https://github.com/cjsergienko/photos.git
cd photos
```

2. **Download and install Claude Code:**
   - Visit [claude.com/claude-code](https://claude.com/claude-code)
   - Download for your operating system
   - Install and open Claude Code in this directory

3. **Ask Claude Code to run the application:**
```
Run the photo restoration app
```

## Manual Installation (Advanced)

If you prefer to set up manually:

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM (recommended for AI models)
- 2GB+ disk space (for AI model downloads)

### Setup Steps

1. Clone the repository and navigate to it
2. Create a virtual environment: `python3 -m venv venv`
3. Activate it: `source venv/bin/activate` (On Windows: `venv\Scripts\activate`)
4. Install core dependencies:
```bash
pip install flask opencv-python-headless numpy werkzeug
```
5. Install AI dependencies (optional, for Fine Tuning):
```bash
pip install torch torchvision torchaudio
pip install basicsr realesrgan gfpgan
pip install deoldify fastai==1.0.61
```
6. Run the app: `python app.py`
7. Open http://localhost:8080 in your browser

**Note:** AI models (~750MB total) are downloaded automatically on first use.

## How to Use

1. Open your browser to **http://localhost:8080**

2. **Upload your old photo:**
   - Click the upload area or drag and drop your image
   - Supported formats: PNG, JPG, JPEG, GIF, BMP
   - Max file size: 16MB

3. **Click "Upload and Restore"**
   - AI will process your photo (5-10 seconds after model download)
   - First run downloads the Real-ESRGAN model (~11MB)

4. **Fine-tune with live sliders:**
   - Adjust Contrast, Noise Reduction, Sharpness, Saturation, and Brightness
   - See changes instantly in your browser
   - No waiting, no server processing

5. **Download your final result!**

## Adjustment Tips

- **Contrast (50 = balanced)**: Lower for subtle look, higher for dramatic enhancement
- **Noise Reduction (50 = moderate)**: Increase for grainy photos, decrease to preserve texture
- **Sharpness (50 = moderate)**: Boost for blurry photos, but avoid over-sharpening
- **Saturation (50 = normal)**: 0 = grayscale, 100 = vivid colors
- **Brightness (50 = normal)**: Adjust for dark or overexposed photos

## Tips for Best Results

1. **Scan at high resolution**: The better your input, the better the AI output
2. **Let AI do the heavy lifting**: The neural network handles upscaling and enhancement
3. **Use sliders for fine-tuning**: Adjust the AI result to your taste
4. **Keep originals**: Always preserve your original scans
5. **Reset and experiment**: Use the Reset button to try different slider combinations
6. **Compare side-by-side**: View original vs enhanced to find the perfect settings

## Project Structure

```
photos/
├── app.py                   # Flask web application (AI processing)
├── install.sh              # macOS installation script
├── requirements.txt         # Python dependencies
├── templates/
│   ├── base.html           # Base template with sidebar navigation
│   ├── batch.html          # Batch enhancement interface
│   └── fine_tuning.html    # Fine tuning with AI restoration
├── static/
│   └── imageProcessor.js   # Client-side image adjustment engine
├── uploads/                # Uploaded photos (created automatically)
├── results/                # AI-restored photos (created automatically)
└── README.md               # This file
```

## Troubleshooting

### First Run Is Slow
- The Real-ESRGAN model downloads automatically (~11MB)
- This only happens once
- Subsequent uploads are much faster (5-10 seconds)

### Slow Internet Connection
- Model download may take 30-60 seconds on slow connections
- Be patient on the first upload
- The model is cached for future use

### Sliders Not Responding
- Make sure you've completed the upload first
- The AI must process the image before sliders become active
- Refresh the page and try again if stuck

### Results Look Over-Processed
- Use the live sliders to reduce enhancement intensity
- Lower Contrast, Sharpness, or Saturation
- Try the Reset button to start from default settings

## Technologies Used

### Server-Side (AI Processing)
- **Real-ESRGAN**: State-of-the-art neural network for image super-resolution
- **GFPGAN**: Deep learning face restoration
- **DeOldify**: AI colorization for black & white photos
- **PyTorch**: Deep learning framework
- **Flask**: Lightweight web framework
- **OpenCV**: Computer vision library (CLAHE, denoising, inpainting)
- **NumPy**: Numerical computing

### Client-Side (Live Adjustments)
- **Canvas API**: Browser-based real-time image manipulation
- **JavaScript**: Client-side image processing engine
- **HTML5**: Modern web interface with sidebar navigation

## Image Processing Techniques

### AI Enhancement (Real-ESRGAN)
- Deep neural network trained on millions of images
- 2x upscaling with quality enhancement
- Artifact reduction and detail restoration
- Learned patterns for realistic image enhancement

### Live Adjustments (Canvas API)
- **Brightness**: Linear RGB adjustment
- **Contrast**: Contrast curve around midpoint (128)
- **Saturation**: HSV color space manipulation
- **Sharpening**: CSS filter-based enhancement (optimized for performance)

## Deploy to Render (Free)

This app is ready to deploy to Render's free tier:

1. **Fork or use this repository**
   - Already configured with `render.yaml`

2. **Sign up for Render**
   - Visit [render.com](https://render.com) and sign up
   - Connect your GitHub account

3. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will auto-detect the `render.yaml` configuration

4. **Deploy**
   - Click "Create Web Service"
   - Render will automatically:
     - Install dependencies
     - Start the app with gunicorn
     - Provide a public URL (e.g., `https://your-app.onrender.com`)

5. **Access your app**
   - Your app will be live at the provided URL
   - First request may take 30-60 seconds (free tier sleeps when inactive)

**Note**: Free tier sleeps after 15 minutes of inactivity. The app will wake up automatically when accessed but may take 30-60 seconds to respond.

## References

- [Real-ESRGAN Paper](https://github.com/xinntao/Real-ESRGAN)
- [GFPGAN Project](https://github.com/TencentARC/GFPGAN)
- [DeOldify Project](https://github.com/jantic/DeOldify)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Canvas API Reference](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API)

## License

This project is open source. OpenCV is licensed under Apache 2.0.
