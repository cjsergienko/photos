import os
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def enhance_photo(img, contrast=50, denoise=50, sharpen=50, saturation=50, brightness=50):
    """
    Enhance old photos using OpenCV with customizable parameters

    Parameters:
    - contrast: 0-100 (adaptive contrast enhancement)
    - denoise: 0-100 (noise reduction strength)
    - sharpen: 0-100 (sharpening strength)
    - saturation: 0-100 (color saturation boost)
    - brightness: 0-100 (brightness adjustment)
    """
    result = img.copy()

    # Brightness adjustment
    if brightness != 50:
        brightness_factor = (brightness - 50) / 50.0  # -1 to 1
        result = cv2.convertScaleAbs(result, alpha=1.0, beta=brightness_factor * 30)

    # Contrast enhancement using CLAHE
    if contrast > 0:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clip_limit = (contrast / 50.0) * 2.0  # 0 to 4
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)

        enhanced_lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Denoise
    if denoise > 0:
        h = int((denoise / 100.0) * 15)  # 0 to 15
        if h > 0:
            result = cv2.fastNlMeansDenoisingColored(result, None, h, h, 7, 21)

    # Sharpen
    if sharpen > 0:
        sharpen_strength = sharpen / 100.0
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(result, -1, kernel * sharpen_strength)
        result = cv2.addWeighted(result, 1 - sharpen_strength * 0.5, sharpened, sharpen_strength * 0.5, 0)

    # Color saturation
    if saturation != 50:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        saturation_factor = saturation / 50.0  # 0 to 2
        hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return result


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and restoration"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get parameters from sliders (0-100 range)
        contrast = int(request.form.get('contrast', 50))
        denoise = int(request.form.get('denoise', 50))
        sharpen = int(request.form.get('sharpen', 50))
        saturation = int(request.form.get('saturation', 50))
        brightness = int(request.form.get('brightness', 50))

        # Read and enhance image
        img = cv2.imread(filepath)
        enhanced = enhance_photo(img, contrast, denoise, sharpen, saturation, brightness)

        # Save result
        result_path = os.path.join(app.config['RESULT_FOLDER'], f'restored_{filename}')
        cv2.imwrite(result_path, enhanced)

        return jsonify({
            'success': True,
            'original': f'/uploads/{filename}',
            'restored': f'/results/restored_{filename}',
            'message': 'Photo restored successfully!'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


@app.route('/results/<filename>')
def result_file(filename):
    """Serve result files"""
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename))


@app.route('/download/<filename>')
def download_file(filename):
    """Download result file"""
    return send_file(
        os.path.join(app.config['RESULT_FOLDER'], filename),
        as_attachment=True,
        download_name=filename
    )


if __name__ == '__main__':
    print("=" * 60)
    print("Photo Restoration Web App - Starting...")
    print("=" * 60)
    print("\nOpen your browser and navigate to:")
    print("  http://localhost:8080")
    print("\nUsing fast OpenCV-based enhancement")
    print("Processing takes ~2-5 seconds per photo")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=8080)
