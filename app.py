import os
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path
import torch
from deoldify import device
from deoldify.device_id import DeviceId
from deoldify.visualize import get_image_colorizer
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Initialize the colorizer (will be loaded on first use)
colorizer = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_colorizer():
    """Lazy load the colorizer model"""
    global colorizer
    if colorizer is None:
        # Set device to CPU or GPU
        device.set(device=DeviceId.GPU0 if torch.cuda.is_available() else DeviceId.CPU)
        colorizer = get_image_colorizer(artistic=True)
    return colorizer


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

        # Get render factor from request (default to 35)
        render_factor = int(request.form.get('render_factor', 35))
        render_factor = max(10, min(45, render_factor))  # Clamp between 10 and 45

        # Restore the photo
        model = get_colorizer()
        result_path = os.path.join(app.config['RESULT_FOLDER'], f'restored_{filename}')

        # Run the restoration
        result = model.get_transformed_image(
            path=filepath,
            render_factor=render_factor,
            post_process=True
        )

        # Save result
        result.save(result_path)

        return jsonify({
            'success': True,
            'original': f'/uploads/{filename}',
            'restored': f'/results/restored_{filename}',
            'message': 'Photo restored successfully!'
        })

    except Exception as e:
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
    print("\nNote: First restoration may take longer as the model downloads...")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=8080)
