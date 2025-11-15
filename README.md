# Photo Restoration Web App

AI-powered web application for restoring and enhancing old photos using Real-ESRGAN neural network.

## Features

- **AI-Powered Restoration**: Uses Real-ESRGAN x2 for fast image enhancement
- **Web Interface**: Simple, user-friendly drag-and-drop interface
- **Adjustable Quality**: Control the quality/speed tradeoff with the render factor slider
- **Side-by-Side Comparison**: View original and restored photos together
- **Download Results**: Save your restored photos locally

## How It Works

**Real-ESRGAN x2**: Enhances image quality through super-resolution
- Reduces noise and artifacts
- Sharpens details
- Upscales images by 2x while preserving quality
- Fast model optimized for quick processing (~11MB download)

## Quick Start (Recommended)

The easiest way to run this application is using Claude Code:

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

Claude Code will automatically:
- Create a virtual environment
- Install all dependencies
- Start the web server
- Open the app at http://localhost:8080

**Note**: On first photo upload, the app will download the AI model (~11MB) which takes 10-30 seconds.

## Manual Installation (Advanced)

If you prefer to set up manually:

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM recommended
- Optional: NVIDIA GPU with CUDA for faster processing

### Setup Steps

1. Clone the repository and navigate to it
2. Create a virtual environment: `python3 -m venv venv`
3. Activate it: `source venv/bin/activate` (On Windows: `venv\Scripts\activate`)
4. Install dependencies: `pip install -r requirements.txt`
5. Run the app: `python app.py`
6. Open http://localhost:8080 in your browser

**Note**: First time setup may take several minutes as it downloads PyTorch and other large dependencies.

## How to Use

1. Open your browser to **http://localhost:8080**

2. **Upload your old photo:**
   - Click the upload area or drag and drop your image
   - Supported formats: PNG, JPG, JPEG, GIF, BMP
   - Max file size: 16MB

3. **Click "Upload and Restore"** and wait for processing

4. **Download your restored photo!**

## Quality Settings Explained

The **render_factor** parameter controls the quality of the restoration:

- **Lower values (10-20)**: Faster processing but may miss fine details
- **Medium values (25-35)**: Good balance between quality and speed
- **Higher values (40-45)**: Best quality but slower, especially for large images

## Tips for Best Results

1. **Scan at high resolution**: The better your input, the better the output
2. **Start with medium quality**: Test with render_factor=35 first
3. **Keep originals**: Always preserve your original scans
4. **GPU acceleration**: If you have an NVIDIA GPU, the processing will be much faster
5. **Manual touchup**: Consider manual editing for important areas after AI restoration

## Project Structure

```
photos/
├── app.py              # Flask web application
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html     # Web interface
├── uploads/           # Uploaded photos (created automatically)
├── results/           # Restored photos (created automatically)
└── README.md          # This file
```

## Troubleshooting

### Model Download Issues
On first run, DeOldify will download pre-trained models (~300MB). If this fails:
- Check your internet connection
- Ensure you have enough disk space
- Try running again

### Out of Memory Errors
If you get memory errors:
- Reduce the render_factor
- Resize your image before uploading
- Close other applications

### Slow Processing
- First restoration is always slower (model initialization)
- Large images take longer
- Consider using GPU if available
- Lower the render_factor

## Technologies Used

- **Real-ESRGAN**: Super-resolution model for image enhancement
- **Flask**: Web framework
- **PyTorch**: Deep learning framework
- **OpenCV**: Image processing library

## References

- [Real-ESRGAN GitHub](https://github.com/xinntao/Real-ESRGAN)
- [Real-ESRGAN Paper](https://arxiv.org/abs/2107.10833)

## License

This project uses Real-ESRGAN which is BSD-3-Clause licensed. See individual component licenses for details.
