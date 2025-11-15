import os
import cv2
import numpy as np
from app import fast_enhance

# Create restored folder if it doesn't exist
os.makedirs('restored', exist_ok=True)

# Get all files from sources folder
sources_dir = 'sources'
files = sorted([f for f in os.listdir(sources_dir) if f.endswith('.jpg')],
               key=lambda x: int(x.split('.')[0]))

total = len(files)
print(f"Found {total} images to process")
print("=" * 60)

processed = 0
errors = 0

for i, filename in enumerate(files, 1):
    try:
        # Read image
        input_path = os.path.join(sources_dir, filename)
        img = cv2.imread(input_path)

        if img is None:
            print(f"❌ Error reading {filename}")
            errors += 1
            continue

        # Enhance image
        enhanced = fast_enhance(img)

        # Save to restored folder with same filename
        output_path = os.path.join('restored', filename)
        success = cv2.imwrite(output_path, enhanced)

        if success:
            processed += 1
            print(f"✅ [{i}/{total}] Processed {filename}")
        else:
            print(f"❌ [{i}/{total}] Failed to save {filename}")
            errors += 1

    except Exception as e:
        print(f"❌ [{i}/{total}] Error processing {filename}: {str(e)}")
        errors += 1

print("=" * 60)
print(f"✅ Successfully processed: {processed} images")
if errors > 0:
    print(f"❌ Errors: {errors}")
print(f"📁 Saved to: restored/")
