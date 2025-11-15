# Photo Restoration Web App

Web application for restoring and enhancing old photos using advanced image processing.

## Features

- **Fast Photo Enhancement**: Uses OpenCV image processing (2-5 seconds per photo!)
- **Web Interface**: Simple, user-friendly drag-and-drop interface
- **Adjustable Quality**: Control enhancement intensity with the quality slider
- **Side-by-Side Comparison**: View original and restored photos together
- **Download Results**: Save your restored photos locally
- **No Downloads Required**: Works instantly, no model downloads needed

## How It Works

**OpenCV-based Enhancement Pipeline:**
- **Adaptive Contrast**: CLAHE algorithm improves brightness and contrast
- **Noise Reduction**: Advanced denoising while preserving edges
- **Sharpening**: Enhances details and textures
- **Color Enhancement**: Boosts saturation for more vibrant photos
- **Super Fast**: 2-5 seconds per photo (vs 5+ minutes with AI models)

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
- Install all dependencies (Flask, OpenCV, NumPy)
- Start the web server
- Open the app at http://localhost:8080

## Manual Installation (Advanced)

If you prefer to set up manually:

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 1GB+ RAM

### Setup Steps

1. Clone the repository and navigate to it
2. Create a virtual environment: `python3 -m venv venv`
3. Activate it: `source venv/bin/activate` (On Windows: `venv\Scripts\activate`)
4. Install dependencies: `pip install -r requirements.txt` (takes ~30 seconds)
5. Run the app: `python app.py`
6. Open http://localhost:8080 in your browser

## How to Use

1. Open your browser to **http://localhost:8080**

2. **Upload your old photo:**
   - Click the upload area or drag and drop your image
   - Supported formats: PNG, JPG, JPEG, GIF, BMP
   - Max file size: 16MB

3. **Click "Upload and Restore"** and wait for processing

4. **Download your restored photo!**

## Quality Settings Explained

The quality slider controls the enhancement intensity:

- **Lower values (10-20)**: Subtle enhancement, preserves original look
- **Medium values (25-35)**: Balanced enhancement (recommended)
- **Higher values (40-45)**: Maximum enhancement, most dramatic results

## Tips for Best Results

1. **Scan at high resolution**: The better your input, the better the output
2. **Start with medium quality**: Use slider value 30-35 for balanced results
3. **Keep originals**: Always preserve your original scans
4. **Adjust intensity**: If results look too processed, lower the quality slider
5. **Compare before/after**: Use the side-by-side view to find the perfect setting

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

### Slow Processing
If processing takes longer than 10 seconds:
- Large images (>10MB) take longer
- Try reducing image resolution before uploading
- Close other applications to free up CPU

### Results Look Over-Processed
- Lower the quality slider (try 20-25)
- Some photos look better with subtle enhancement

### No Visible Difference
- Try increasing the quality slider (try 40-45)
- Very dark or very bright photos may need manual adjustment first

## Technologies Used

- **OpenCV**: Advanced computer vision and image processing library
- **Flask**: Lightweight web framework
- **NumPy**: Numerical computing library

## Image Processing Techniques

- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Improves local contrast
- **Non-Local Means Denoising**: Reduces noise while preserving edges
- **Unsharp Masking**: Enhances fine details and sharpness
- **HSV Color Enhancement**: Boosts color saturation

## References

- [OpenCV Documentation](https://docs.opencv.org/)
- [CLAHE Algorithm](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization)

## License

This project is open source. OpenCV is licensed under Apache 2.0.
