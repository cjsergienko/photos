import os
import zipfile
import io
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max total upload size

# Use /tmp for file storage on Render (read-only filesystem)
base_dir = '/tmp' if os.environ.get('RENDER') else '.'
app.config['UPLOAD_FOLDER'] = os.path.join(base_dir, 'uploads')
app.config['RESULT_FOLDER'] = os.path.join(base_dir, 'results')

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# DeOldify colorizer (lazy loaded)
deoldify_colorizer = None
# Real-ESRGAN upsampler (lazy loaded)
realesrgan_upsampler = None
# GFPGAN face enhancer (lazy loaded)
gfpgan_enhancer = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def remove_artifacts(img):
    """
    Remove bright dust spots, scratches, and lines from old photos.
    Only targets artifacts without degrading flat areas like sky or snow.
    """
    result = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Detect dust spots using top-hat transform
    # Focus on medium-sized kernel to catch dust spots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    # Top-hat transform reveals bright spots on darker/uniform background
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    # Higher threshold to only catch obvious bright spots
    _, artifact_mask = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)

    # 2. Filter out large areas and keep only small spots
    contours, _ = cv2.findContours(artifact_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(artifact_mask)

    for contour in contours:
        area = cv2.contourArea(contour)
        # Keep only small to medium spots (dust spots are typically 5-200 pixels)
        if 5 < area < 200:
            cv2.drawContours(filtered_mask, [contour], -1, 255, -1)

    # 3. Dilate slightly to cover edges of spots
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filtered_mask = cv2.dilate(filtered_mask, kernel_dilate, iterations=1)

    # 4. Safety check - don't process if too many pixels detected (something's wrong)
    num_pixels = np.sum(filtered_mask > 0)
    if num_pixels > 50000:
        print(f"Warning: Too many artifact pixels ({num_pixels}), skipping to avoid damage")
        return result

    # 5. Inpaint only the artifact areas
    if num_pixels > 0:
        print(f"Found {num_pixels} artifact pixels to remove...")
        result = cv2.inpaint(result, filtered_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    else:
        print("No artifacts detected")

    return result


def fast_enhance(img):
    """
    First step: Fast OpenCV-based enhancement for Batch Enhancement
    This runs once on the server when the photo is uploaded
    Keeps it simple and fast - no heavy artifact removal
    """
    # Apply stronger enhancement - more contrast, saturation, and sharpness
    # But NO heavy denoising or artifact removal - keep it fast
    result = enhance_photo(img, contrast=60, denoise=15, sharpen=55, saturation=65, brightness=52)
    return result


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

    # Sharpen (using gentler unsharp mask)
    if sharpen > 0:
        sharpen_strength = sharpen / 100.0
        # Use Gaussian blur for unsharp mask (smoother, less noise)
        blurred = cv2.GaussianBlur(result, (0, 0), 3)
        result = cv2.addWeighted(result, 1 + sharpen_strength * 0.5, blurred, -sharpen_strength * 0.5, 0)

    # Color saturation
    if saturation != 50:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        saturation_factor = saturation / 50.0  # 0 to 2
        hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return result


def get_deoldify_colorizer():
    """Lazy load DeOldify colorizer"""
    global deoldify_colorizer
    if deoldify_colorizer is None:
        try:
            from deoldify import device
            from deoldify.device_id import DeviceId
            device.set(device=DeviceId.CPU)

            from deoldify.visualize import get_image_colorizer
            deoldify_colorizer = get_image_colorizer(artistic=True)
            print("DeOldify colorizer loaded successfully!")
        except ImportError as e:
            print(f"DeOldify not available: {e}")
            print("Install with: pip install deoldify")
            return None
        except Exception as e:
            print(f"Error loading DeOldify: {e}")
            return None
    return deoldify_colorizer


def get_realesrgan_upsampler():
    """Lazy load Real-ESRGAN upsampler for AI restoration"""
    global realesrgan_upsampler
    if realesrgan_upsampler is None:
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            # Check for pre-downloaded model first
            local_model_path = os.path.expanduser("~/.cache/realesrgan/RealESRGAN_x4plus.pth")
            if os.path.exists(local_model_path):
                model_path = local_model_path
                print(f"Using pre-downloaded Real-ESRGAN model from {local_model_path}")
            else:
                model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
                print("Downloading Real-ESRGAN model (this will be cached)...")

            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            realesrgan_upsampler = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=400,  # Use tiling for large images
                tile_pad=10,
                pre_pad=0,
                half=False,  # Use full precision for CPU
                device='cpu'
            )
            print("Real-ESRGAN upsampler loaded successfully!")
        except ImportError as e:
            print(f"Real-ESRGAN not available: {e}")
            print("Install with: pip install realesrgan")
            return None
        except Exception as e:
            print(f"Error loading Real-ESRGAN: {e}")
            import traceback
            traceback.print_exc()
            return None
    return realesrgan_upsampler


def get_gfpgan_enhancer():
    """Lazy load GFPGAN for face restoration"""
    global gfpgan_enhancer
    if gfpgan_enhancer is None:
        try:
            from gfpgan import GFPGANer

            # Check for pre-downloaded model first
            local_model_path = os.path.expanduser("~/.cache/gfpgan/weights/GFPGANv1.3.pth")
            if os.path.exists(local_model_path):
                model_path = local_model_path
                print(f"Using pre-downloaded GFPGAN model from {local_model_path}")
            else:
                model_path = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
                print("Downloading GFPGAN model (this will be cached)...")

            gfpgan_enhancer = GFPGANer(
                model_path=model_path,
                upscale=1,  # No upscaling - just enhance quality (faster)
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None  # Don't use Real-ESRGAN for background (much faster)
            )
            print("GFPGAN face enhancer loaded successfully!")
        except ImportError as e:
            print(f"GFPGAN not available: {e}")
            print("Install with: pip install gfpgan")
            return None
        except Exception as e:
            print(f"Error loading GFPGAN: {e}")
            import traceback
            traceback.print_exc()
            return None
    return gfpgan_enhancer


@app.route('/')
def index():
    """Render the batch enhancement page (main page)"""
    return render_template('batch.html', active_page='batch')


@app.route('/fine-tuning')
def fine_tuning():
    """Render the fine tuning page"""
    return render_template('fine_tuning.html', active_page='fine_tuning')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle single or batch file upload and restoration"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')

        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400

        results = []
        errors = []

        for file in files:
            if file.filename == '':
                continue

            if not allowed_file(file.filename):
                errors.append(f'{file.filename}: Invalid file type')
                continue

            try:
                # Save uploaded file
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Step 1: Fast OpenCV enhancement (runs once, server-side)
                print(f"Processing {filename} with OpenCV...")
                img = cv2.imread(filepath)

                if img is None:
                    errors.append(f'{filename}: Failed to read image')
                    continue

                enhanced = fast_enhance(img)

                # Save enhanced version
                result_path = os.path.join(app.config['RESULT_FOLDER'], f'enhanced_{filename}')
                success = cv2.imwrite(result_path, enhanced)

                if not success:
                    errors.append(f'{filename}: Failed to save enhanced image')
                    continue

                print(f"Enhancement complete for {filename}!")

                results.append({
                    'filename': filename,
                    'original': f'/uploads/{filename}',
                    'enhanced': f'/results/enhanced_{filename}'
                })

            except Exception as e:
                errors.append(f'{file.filename}: {str(e)}')
                continue

        if not results:
            return jsonify({'error': 'No images were processed successfully', 'errors': errors}), 400

        return jsonify({
            'success': True,
            'results': results,
            'errors': errors if errors else None,
            'count': len(results),
            'message': f'Successfully enhanced {len(results)} photo(s)!'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing images: {str(e)}'}), 500


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


@app.route('/download-all', methods=['POST'])
def download_all():
    """Download all enhanced images as a ZIP file"""
    try:
        data = request.get_json()
        filenames = data.get('filenames', [])

        if not filenames:
            return jsonify({'error': 'No filenames provided'}), 400

        # Create a ZIP file in memory
        memory_file = io.BytesIO()

        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for filename in filenames:
                enhanced_filename = f'enhanced_{filename}'
                filepath = os.path.join(app.config['RESULT_FOLDER'], enhanced_filename)

                if os.path.exists(filepath):
                    # Add file to zip with a clean name
                    zf.write(filepath, enhanced_filename)

        memory_file.seek(0)

        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='enhanced_photos.zip'
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error creating ZIP file: {str(e)}'}), 500


@app.route('/fine-tune', methods=['POST'])
def fine_tune():
    """Handle single photo fine tuning with AI restoration"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        use_deoldify = request.form.get('deoldify', 'true').lower() == 'true'
        use_ai_restoration = request.form.get('ai_restoration', 'false').lower() == 'true'
        use_enhance = request.form.get('enhance', 'true').lower() == 'true'
        use_noise_reduction = request.form.get('noise_reduction', 'true').lower() == 'true'

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result_filename = f'finetuned_{filename}'
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

        # Load image
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Failed to read image'}), 400

        # Process with DeOldify if requested (colorization)
        if use_deoldify:
            colorizer = get_deoldify_colorizer()
            if colorizer:
                try:
                    print(f"Applying DeOldify colorization to {filename}...")
                    # DeOldify expects the file path
                    result_image = colorizer.get_transformed_image(
                        path=filepath,
                        render_factor=35
                    )
                    # Convert PIL Image to OpenCV format for further processing
                    img = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
                    print("DeOldify colorization complete!")
                except Exception as e:
                    print(f"DeOldify processing failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("DeOldify not available, skipping colorization")

        # Apply AI restoration (Real-ESRGAN + GFPGAN) if requested
        if use_ai_restoration:
            try:
                import time
                start_time = time.time()

                original_size = (img.shape[1], img.shape[0])
                print(f"Applying AI restoration to {filename}...")
                print(f"Original image size: {original_size[0]}x{original_size[1]}")

                # Downscale very large images for faster processing
                max_dimension = 1024  # Process at max 1024px for Real-ESRGAN (it will 4x upscale)
                scale_factor = 1.0
                processing_img = img

                if max(original_size) > max_dimension:
                    scale_factor = max_dimension / max(original_size)
                    new_width = int(original_size[0] * scale_factor)
                    new_height = int(original_size[1] * scale_factor)
                    processing_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    print(f"Downscaled to {new_width}x{new_height} for AI processing...")

                # First try GFPGAN for face enhancement
                enhancer = get_gfpgan_enhancer()
                faces_enhanced = False
                if enhancer:
                    print("Running GFPGAN face detection and enhancement...")
                    cropped_faces, restored_faces, restored_img = enhancer.enhance(
                        processing_img,
                        has_aligned=False,
                        only_center_face=False,
                        paste_back=True,
                        weight=0.5
                    )

                    if restored_img is not None and len(cropped_faces) > 0:
                        processing_img = restored_img
                        faces_enhanced = True
                        print(f"Enhanced {len(cropped_faces)} face(s) with GFPGAN!")
                    else:
                        print("No faces detected, skipping GFPGAN...")

                # Apply Real-ESRGAN for full image enhancement
                upsampler = get_realesrgan_upsampler()
                if upsampler:
                    print("Running Real-ESRGAN image enhancement...")
                    output, _ = upsampler.enhance(processing_img, outscale=2)  # 2x upscale instead of 4x
                    if output is not None:
                        processing_img = output
                        print(f"Real-ESRGAN enhanced to {processing_img.shape[1]}x{processing_img.shape[0]}")
                else:
                    print("Real-ESRGAN not available, skipping...")

                # Resize back to original dimensions
                if processing_img.shape[1] != original_size[0] or processing_img.shape[0] != original_size[1]:
                    print(f"Resizing to original size {original_size[0]}x{original_size[1]}...")
                    processing_img = cv2.resize(processing_img, original_size, interpolation=cv2.INTER_LANCZOS4)

                img = processing_img
                elapsed = time.time() - start_time
                print(f"AI restoration complete! (took {elapsed:.1f}s)")

            except Exception as e:
                print(f"AI restoration failed: {e}")
                import traceback
                traceback.print_exc()

        # Apply noise reduction if requested (removes dust spots/scratches)
        if use_noise_reduction:
            print(f"Applying noise reduction to {filename}...")
            img = remove_artifacts(img)

        # Apply standard enhancement if requested
        if use_enhance:
            print(f"Applying standard enhancement to {filename}...")
            img = fast_enhance(img)

        # Save result
        success = cv2.imwrite(result_path, img)
        if not success:
            return jsonify({'error': 'Failed to save result'}), 500

        print(f"Fine tuning complete for {filename}!")

        return jsonify({
            'success': True,
            'result_url': f'/results/{result_filename}',
            'result_filename': result_filename,
            'message': 'Photo processed successfully!'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500


if __name__ == '__main__':
    import os

    # Get port from environment variable (for production) or use 8080 for local dev
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_ENV') != 'production'

    print("=" * 60)
    print("Photo Restoration Web App - Starting...")
    print("=" * 60)
    print(f"\nServer running on port: {port}")
    if debug:
        print("Open your browser and navigate to:")
        print(f"  http://localhost:{port}")
    print("\n🎨 Two-Step Enhancement:")
    print("  1. Fast OpenCV enhancement (server-side)")
    print("  2. Live slider adjustments (client-side)")
    print("\n⚡ Processing: ~2-5 seconds per photo")
    print("   No model downloads needed!")
    print("=" * 60)

    app.run(debug=debug, host='0.0.0.0', port=port)
