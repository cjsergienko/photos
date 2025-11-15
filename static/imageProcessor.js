// Client-side real-time image processing using Canvas API

class ImageProcessor {
    constructor(canvas, img) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d', { willReadFrequently: true });
        this.originalImage = img;

        // Set canvas size to match image
        this.canvas.width = img.width;
        this.canvas.height = img.height;

        // Draw original image
        this.ctx.drawImage(img, 0, 0);

        // Store original image data
        this.originalImageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    }

    // Apply all adjustments
    applyAdjustments(contrast, denoise, sharpen, saturation, brightness) {
        // Start with original image data
        let imageData = new ImageData(
            new Uint8ClampedArray(this.originalImageData.data),
            this.originalImageData.width,
            this.originalImageData.height
        );

        // Apply brightness
        imageData = this.adjustBrightness(imageData, brightness);

        // Apply contrast
        imageData = this.adjustContrast(imageData, contrast);

        // Apply saturation
        imageData = this.adjustSaturation(imageData, saturation);

        // Apply sharpening (simplified - uses CSS filter for performance)
        this.ctx.putImageData(imageData, 0, 0);

        // Apply sharpen filter using CSS (faster than pixel manipulation)
        if (sharpen > 50) {
            const sharpenAmount = (sharpen - 50) / 50 * 2; // 0 to 2
            this.ctx.filter = `contrast(${1 + sharpenAmount * 0.2})`;
            this.ctx.drawImage(this.canvas, 0, 0);
            this.ctx.filter = 'none';
        }
    }

    adjustBrightness(imageData, value) {
        const data = imageData.data;
        const factor = (value - 50) / 50.0 * 30; // -30 to +30

        for (let i = 0; i < data.length; i += 4) {
            data[i] = Math.max(0, Math.min(255, data[i] + factor));     // R
            data[i + 1] = Math.max(0, Math.min(255, data[i + 1] + factor)); // G
            data[i + 2] = Math.max(0, Math.min(255, data[i + 2] + factor)); // B
        }

        return imageData;
    }

    adjustContrast(imageData, value) {
        const data = imageData.data;
        const factor = (value / 50.0) * 2; // 0 to 4

        for (let i = 0; i < data.length; i += 4) {
            data[i] = Math.max(0, Math.min(255, factor * (data[i] - 128) + 128));     // R
            data[i + 1] = Math.max(0, Math.min(255, factor * (data[i + 1] - 128) + 128)); // G
            data[i + 2] = Math.max(0, Math.min(255, factor * (data[i + 2] - 128) + 128)); // B
        }

        return imageData;
    }

    adjustSaturation(imageData, value) {
        const data = imageData.data;
        const factor = value / 50.0; // 0 to 2

        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];

            // Convert to grayscale
            const gray = 0.299 * r + 0.587 * g + 0.114 * b;

            // Interpolate between grayscale and original based on factor
            data[i] = Math.max(0, Math.min(255, gray + factor * (r - gray)));
            data[i + 1] = Math.max(0, Math.min(255, gray + factor * (g - gray)));
            data[i + 2] = Math.max(0, Math.min(255, gray + factor * (b - gray)));
        }

        return imageData;
    }

    // Get the current canvas as a blob for downloading
    getBlob(callback) {
        this.canvas.toBlob(callback, 'image/jpeg', 0.95);
    }
}
