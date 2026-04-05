import os 
import numpy as np 

from PIL import Image 
import cv2

# =========================
# INTERPOLATION FUNCTION
# =========================

def generate_interpolated(input_dir, output_dir, alpha, mask=False, threshold_value=127):
    """
    Generate interpolated slices between consecutive images in a directory.
    
    Parameters:
    - input_dir: path containing original slices (images or masks)
    - output_dir: path to save interpolated slices
    - alpha: interpolation factor (0.0 = first slice, 1.0 = next slice)
    - mask: True if images are masks, False for normal images
    - threshold_value: threshold for binary masks
    
    Steps:
    1. Load all images (skipping ones that already have '_5' in filename)
    2. Optionally binarize images if mask=True
    3. Interpolate between consecutive slices
    4. Save interpolated slice with '_5' appended to filename
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- filter files: only images, ignore already interpolated '_5' slices ---
    files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and not os.path.splitext(f)[0].endswith('_5')
    ])

    print(f"Loading {len(files)} files from {input_dir}")

    images = []

    # --- load images ---
    for f in files:
        path = os.path.join(input_dir, f)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print("Failed to read:", path)
            continue

        if mask:
            # Threshold masks to binary
            _, img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

        images.append(img)

    # --- interpolate between consecutive slices ---
    for i in range(len(images) - 1):
        img1 = images[i].astype(np.float32)
        img2 = images[i + 1].astype(np.float32)

        base_name = f"{i+1:05d}"  # Original slice index

        # --- handle empty slices ---
        if np.all(img1 == 0) or np.all(img2 == 0):
            interpolated = np.zeros_like(img1, dtype=np.uint8)
        else:
            # Linear interpolation formula: (1-alpha)*img1 + alpha*img2
            interpolated = (1 - alpha) * img1 + alpha * img2
            interpolated = np.clip(interpolated, 0, 255).astype(np.uint8)

            if mask:
                # Convert interpolated mask to binary 0-255 by tresholding generates mask union
                interpolated = (interpolated > 127).astype(np.uint8) * 255

            # --- save interpolated slice ---
            ext = "png" if mask else "jpg"
            filename = f"{base_name}_5.{ext}"
            out_path = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, interpolated)

    print("Interpolation complete.")

# =========================
# MAIN LOOP
# =========================              


test_train = "test"
cases = list(range(1,20))

# Alternative for training data (commented)
# test_train = "train"
# cases = list(range(20,159))

for i in cases:
    # Base directories for input/output per case
    base_dir_input = fr"C:\Users\20213282\Documents\uni\BEP\codes\data\data_prostate\{test_train}\{i:03d}" 
    base_dir_output = fr"C:\Users\20213282\Documents\uni\BEP\codes\data\data_prostate\{test_train}\{i:03d}"

    # Separate directories for images and masks
    img_dir = os.path.join(base_dir_input, "images") 
    mask_dir = os.path.join(base_dir_input, "masks") 
    output_img_dir = os.path.join(base_dir_output, "images") 
    output_mask_dir = os.path.join(base_dir_output, "masks")
    
    # Generate interpolated images and masks
    generate_interpolated(input_dir = img_dir, output_dir= output_img_dir, alpha=0.5, mask = False ) 
    generate_interpolated(input_dir = mask_dir, output_dir= output_mask_dir, alpha=0.5, mask = True )
