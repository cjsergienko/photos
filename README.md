# Photo Restoration Web App

AI-powered web application for restoring and enhancing old photos using Real-ESRGAN and GFPGAN neural networks.

## Features

- **AI-Powered Restoration**: Uses Real-ESRGAN for image enhancement and GFPGAN for face restoration
- **Web Interface**: Simple, user-friendly drag-and-drop interface
- **Adjustable Quality**: Control the quality/speed tradeoff with the render factor slider
- **Side-by-Side Comparison**: View original and restored photos together
- **Download Results**: Save your restored photos locally

## How It Works

This app uses two powerful AI models:

**Real-ESRGAN**: Enhances image quality through super-resolution
- Reduces noise and artifacts
- Sharpens details
- Upscales images while preserving quality

**GFPGAN**: Specialized in restoring faces
- Enhances facial features
- Fixes degraded face photos
- Works on portraits and group photos

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM recommended
- Optional: NVIDIA GPU with CUDA for faster processing

### Setup

1. Clone or navigate to this repository:
```bash
cd /path/to/photos
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: First time setup may take several minutes as it downloads PyTorch and other large dependencies.

## Usage

1. Start the web server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:8080
```

3. Upload your old photo:
   - Click the upload area or drag and drop your image
   - Supported formats: PNG, JPG, JPEG, GIF, BMP
   - Max file size: 16MB

4. Adjust the quality slider:
   - **10-20**: Fast processing, lower quality (good for testing)
   - **25-35**: Balanced (recommended)
   - **40-45**: Highest quality, slower processing

5. Click "Upload and Restore" and wait for processing

6. Download your restored photo!

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
- **GFPGAN**: GAN-based face restoration model
- **Flask**: Web framework
- **PyTorch**: Deep learning framework
- **OpenCV**: Image processing library

## References

- [Real-ESRGAN GitHub](https://github.com/xinntao/Real-ESRGAN)
- [GFPGAN GitHub](https://github.com/TencentARC/GFPGAN)
- [Real-ESRGAN Paper](https://arxiv.org/abs/2107.10833)
- [GFPGAN Paper](https://arxiv.org/abs/2101.04061)

## License

This project uses Real-ESRGAN and GFPGAN which are BSD-3-Clause licensed. See individual component licenses for details.
