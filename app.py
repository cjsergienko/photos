import os
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path
import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['MODEL_FOLDER'] = 'models'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Initialize the restorer (will be loaded on first use)
upsampler = None
face_enhancer = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_restorer():
    """Lazy load the restoration models"""
    global upsampler, face_enhancer

    if upsampler is None:
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize Real-ESRGAN model
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

        # Initialize GFPGAN for face enhancement
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=4,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler,
            device=device
        )

    return upsampler, face_enhancer


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

        # Get quality setting from request (default to 35)
        quality = int(request.form.get('render_factor', 35))
        use_face_enhance = quality > 25  # Use face enhancement for medium-high quality

        # Read image
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

        # Get restoration models
        upsampler_model, face_enhancer_model = get_restorer()

        # Restore the photo
        result_path = os.path.join(app.config['RESULT_FOLDER'], f'restored_{filename}')

        if use_face_enhance and face_enhancer_model is not None:
            # Use GFPGAN for face enhancement (better for photos with people)
            _, _, output = face_enhancer_model.enhance(
                img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=0.5
            )
        else:
            # Use Real-ESRGAN for general enhancement
            output, _ = upsampler_model.enhance(img, outscale=2)

        # Save result
        cv2.imwrite(result_path, output)

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
    print("\nNote: First restoration may take longer as the model downloads...")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=8080)
