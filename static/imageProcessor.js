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

        // Apply denoising first (before other adjustments)
        if (denoise !== 50) {
            imageData = this.adjustDenoise(imageData, denoise);
        }

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

    adjustDenoise(imageData, value) {
        // Fast noise reduction using smoothing
        // Value 50 = no denoising, <50 = add grain, >50 = reduce noise
        if (value === 50) {
            return imageData;
        }

        const width = imageData.width;
        const height = imageData.height;
        const data = imageData.data;

        if (value > 50) {
            // Noise reduction - fast smoothing approach
            const strength = (value - 50) / 50.0; // 0 to 1
            const output = new Uint8ClampedArray(data.length);

            // Fast 3x3 median-like filter (just average neighbors)
            for (let y = 1; y < height - 1; y++) {
                for (let x = 1; x < width - 1; x++) {
                    const idx = (y * width + x) * 4;

                    // Simple 3x3 average (fast)
                    let r = 0, g = 0, b = 0;
                    for (let dy = -1; dy <= 1; dy++) {
                        for (let dx = -1; dx <= 1; dx++) {
                            const nidx = ((y + dy) * width + (x + dx)) * 4;
                            r += data[nidx];
                            g += data[nidx + 1];
                            b += data[nidx + 2];
                        }
                    }

                    output[idx] = r / 9;
                    output[idx + 1] = g / 9;
                    output[idx + 2] = b / 9;
                    output[idx + 3] = data[idx + 3];
                }
            }

            // Copy edges
            for (let x = 0; x < width; x++) {
                const topIdx = x * 4;
                const bottomIdx = ((height - 1) * width + x) * 4;
                output[topIdx] = data[topIdx];
                output[topIdx + 1] = data[topIdx + 1];
                output[topIdx + 2] = data[topIdx + 2];
                output[topIdx + 3] = data[topIdx + 3];
                output[bottomIdx] = data[bottomIdx];
                output[bottomIdx + 1] = data[bottomIdx + 1];
                output[bottomIdx + 2] = data[bottomIdx + 2];
                output[bottomIdx + 3] = data[bottomIdx + 3];
            }
            for (let y = 0; y < height; y++) {
                const leftIdx = (y * width) * 4;
                const rightIdx = (y * width + width - 1) * 4;
                output[leftIdx] = data[leftIdx];
                output[leftIdx + 1] = data[leftIdx + 1];
                output[leftIdx + 2] = data[leftIdx + 2];
                output[leftIdx + 3] = data[leftIdx + 3];
                output[rightIdx] = data[rightIdx];
                output[rightIdx + 1] = data[rightIdx + 1];
                output[rightIdx + 2] = data[rightIdx + 2];
                output[rightIdx + 3] = data[rightIdx + 3];
            }

            // Blend based on strength
            for (let i = 0; i < data.length; i += 4) {
                data[i] = data[i] * (1 - strength) + output[i] * strength;
                data[i + 1] = data[i + 1] * (1 - strength) + output[i + 1] * strength;
                data[i + 2] = data[i + 2] * (1 - strength) + output[i + 2] * strength;
            }
        } else {
            // Add grain effect (value < 50)
            const grainStrength = (50 - value) / 50.0 * 15; // 0 to 15
            for (let i = 0; i < data.length; i += 4) {
                const grain = (Math.random() - 0.5) * grainStrength;
                data[i] = Math.max(0, Math.min(255, data[i] + grain));
                data[i + 1] = Math.max(0, Math.min(255, data[i + 1] + grain));
                data[i + 2] = Math.max(0, Math.min(255, data[i + 2] + grain));
            }
        }

        return imageData;
    }

    // Get the current canvas as a blob for downloading
    getBlob(callback) {
        this.canvas.toBlob(callback, 'image/jpeg', 0.95);
    }
}
