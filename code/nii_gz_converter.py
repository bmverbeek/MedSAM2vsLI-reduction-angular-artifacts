import sys
print("RUNNING PYTHON:", sys.executable) # Print which Python interpreter is running

import os
import nibabel as nib   # For handling NIfTI medical images
import numpy as np
from PIL import Image   # For saving slices as images
import shutil

# =========================
# PATH CONFIGURATION
# =========================

# Variable which decides what dataset is used
test_train="test"
#test_train="train"

# Root folder containing original NIfTI data
INPUT_ROOT = fr"C:\Users\20213282\Documents\uni\BEP\codes\data\test\prostate158_{test_train}"


# Root folder to save processed output (images, masks, copies of NIfTI) 
OUTPUT_ROOT = fr"C:\Users\20213282\Documents\uni\BEP\codes\data\data_prostate\{test_train}"

def normalize_slice(slice_2d):
    """
    Normalize a 2D MRI slice to the 0-255 range for visualization/saving as image.
    
    Steps:
    1. Convert slice to float32
    2. Min-max normalize to 0-1
    3. Scale to 0-255 and convert to uint8
    """
    slice_2d = slice_2d.astype(np.float32)

    min_val = np.min(slice_2d)
    max_val = np.max(slice_2d)

    if max_val - min_val > 0:
        slice_2d = (slice_2d - min_val) / (max_val - min_val)

    slice_2d = (slice_2d * 255).astype(np.uint8)

    return slice_2d


def process_case(case_folder):
    """
    Process a single patient/case:
    - Normalize MRI slices
    - Convert mask to 0-255
    - Rotate slices for correct orientation
    - Save slices as individual image files
    - Copy original NIfTI files to output folder
    """
    case_id = os.path.basename(case_folder)

    # Paths to original NIfTI files
    t2_path = os.path.join(case_folder, "t2.nii.gz")
    mask_path = os.path.join(case_folder, "t2_anatomy_reader1.nii.gz")

    # Skip case if T2 image is missing
    if not os.path.exists(t2_path):
        print(f"Skipping {case_id}, no t2.nii found")
        return

    print(f"Processing case {case_id}")

    # Set up directory architecture
    image_out = os.path.join(OUTPUT_ROOT, case_id, "images")
    mask_out = os.path.join(OUTPUT_ROOT, case_id, "masks")
    os.makedirs(image_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    # Copy original NIfTI files to output folder
    t2_copy_path = os.path.join(OUTPUT_ROOT, case_id, "t2.nii.gz")
    mask_copy_path = os.path.join(OUTPUT_ROOT, case_id, "t2_anatomy_reader1.nii.gz")

    shutil.copy2(t2_path, t2_copy_path)
    shutil.copy2(mask_path, mask_copy_path)

    # Load NIfTI data as NumPy arrays
    t2_img = nib.load(t2_path).get_fdata()
    mask_img = nib.load(mask_path).get_fdata()

    depth = t2_img.shape[2] # Number of slices along axial plane

    for i in range(depth):

        # Extract individual slices
        t2_slice = t2_img[:, :, i]
        mask_slice = mask_img[:, :, i]

        # Normalize MRI slice to 0-255 and rotate for correct orientation
        t2_slice = normalize_slice(t2_slice)
        t2_slice = np.rot90(t2_slice, k=-1)
        
        # Convert mask to binary 0-255 and rotate
        mask_slice = (mask_slice > 0).astype(np.uint8) * 255
        mask_slice = np.rot90(mask_slice, k=-1)
        
        # Filenames with 5-digit zero padding
        name = f"{i+1:05d}"
        image_file = os.path.join(image_out, name + ".jpg")
        mask_file = os.path.join(mask_out, name + ".png")

        # Save slices as images
        Image.fromarray(t2_slice).save(image_file)
        Image.fromarray(mask_slice).save(mask_file)

    print(f"Saved {depth} slices for case {case_id}")


def main():
    """
    Main function to process all cases in INPUT_ROOT.
    Loops over all folders and applies `process_case`.
    """
    cases = sorted(os.listdir(INPUT_ROOT))  # Sort for consistent order

    for case in cases:

        case_path = os.path.join(INPUT_ROOT, case)

        if os.path.isdir(case_path):    # Only process directories
            process_case(case_path)


if __name__ == "__main__":
    main()
