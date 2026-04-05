import os
import glob
import cv2
import numpy as np
import nibabel as nib
from skimage.transform import resize
from scipy.ndimage import binary_closing
# -----------------------------
# Helper functions
# -----------------------------
def is_empty(img):
    return np.sum(img) == 0

def normalize_image(x):
    x = x - np.min(x)
    x = x / (np.max(x) + 1e-8)
    return (x * 255).astype(np.uint8)

def get_voxel_spacing(case_dir):
    t2_path = os.path.join(case_dir, "t2.nii.gz")
    if not os.path.exists(t2_path):
        raise FileNotFoundError(f"Missing t2.nii.gz in {case_dir}")
    nii = nib.load(t2_path)
    return nii.header.get_zooms()[:3]  # px, py, pz

def pad_slice_to_aspect(img, target_rows, target_cols):
    """
    Pads a 2D mask with zeros to reach target shape (rows, cols).
    """
    rows, cols = img.shape
    pad_top = (target_rows - rows) // 2
    pad_bottom = target_rows - rows - pad_top
    pad_left = (target_cols - cols) // 2
    pad_right = target_cols - cols - pad_left
    
    padded = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    return padded
# -----------------------------
# Compute center in ITK-SNAP coordinates
# -----------------------------
def compute_center_pixel(volume):
    """
    Compute center of mass in pixel coordinates of a 3D numpy volume.
    Returns integer indices (xc, yc, zc) for slicing.
    """
    coords = np.argwhere(volume > 0)
    if coords.size == 0:
        Z, H, W = volume.shape
        return W//2, H//2, Z//2

    z_idx, y_idx, x_idx = coords[:,0], coords[:,1], coords[:,2]
    xc = int(round(x_idx.mean()))
    yc = int(round(y_idx.mean()))
    zc = int(round(z_idx.mean()))
    return xc, yc, zc

# -----------------------------
# Extract center slices
# -----------------------------
def extract_center_views(volume, out_dir, prefix, xc, yc, i, px, py, pz, target_z=155):
    """
    Extract sagittal and coronal views.
    Width is fixed, height scaled using voxel spacing to preserve anatomical proportions.
    """
    os.makedirs(out_dir, exist_ok=True)
    Z, Y, X = volume.shape

    xc = np.clip(xc, 0, X-1)
    yc = np.clip(yc, 0, Y-1)
    # Compute physical sizes
    sag_height_mm = Z * pz
    sag_width_mm  = Y * py
    cor_height_mm = Z * pz
    cor_width_mm  = X * px

    # Determine target cols to preserve aspect ratio
    # Use the maximum physical width as reference
    max_width_mm = max(sag_width_mm, cor_width_mm)

    # Compute padding in pixels
    sag_target_cols = int(round(max_width_mm / py))  # number of columns for sagittal
    cor_target_cols = int(round(max_width_mm / px))  # number of columns for coronal

    # ---------------- Sagittal ----------------
    sag = volume[:, :, xc]           # (Z, Y)
    sag = np.flip(sag, axis=0)      # flip top-bottom
    # Scale height only
    sag_resized = sag
    sag = pad_slice_to_aspect(sag, Z, sag_target_cols)  # pad width (cols)
    sag_resized = cv2.resize(sag, (Y, target_z), interpolation=cv2.INTER_NEAREST)

    # ---------------- Coronal ----------------
    cor = volume[:, yc, :]           # (Z, X)
    cor = np.flip(cor, axis=0)
    cor = np.flip(cor, axis=1)      # optional horizontal flip
    cor_resized = cor
    cor = pad_slice_to_aspect(cor, Z, cor_target_cols)  # pad width (cols)
    cor_resized = cv2.resize(cor, (X, target_z) , interpolation=cv2.INTER_NEAREST)

    # ---------------- Save ----------------
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_sagittal_{i:03d}.png"), sag_resized)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_coronal_{i:03d}.png"), cor_resized)

# -----------------------------
# Build volumes from 2D slices with spacing
# -----------------------------
def build_spaced_volumes(mask_dir, sam2mask_dir, px, pz, target_z=None, Quantative=False):
    """
    Build 3D volumes from axial slices.
    If target_z is provided, the final volume is resampled along Z to match it (to match NIfTI).
    """
    linear_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    sam2_files = sorted(glob.glob(os.path.join(sam2mask_dir, "*.png")))

    org_files = sorted([f for f in linear_files if "_5" not in f and "_5S" not in f])
    lin_interp_files = {os.path.basename(f).replace(".png", ""): f for f in linear_files if "_5" in f and "_5S" not in f}
    sam2_dict = {os.path.basename(f).replace(".png", ""): f for f in sam2_files}

    vol_org, vol_linear, vol_sam2 = [], [], []
    non_empty_indices = [idx for idx, f in enumerate(org_files) 
                     if not is_empty(cv2.imread(f, cv2.IMREAD_GRAYSCALE))]


    if len(non_empty_indices) == 0:
        # All slices are empty, nothing to do
        first_non_empty_idx = 0
        last_non_empty_idx = 0
    else:
        first_non_empty_idx = non_empty_indices[0]
        last_non_empty_idx = non_empty_indices[-1]

    empty_slice = cv2.imread(org_files[0], cv2.IMREAD_GRAYSCALE) * 0
    for _ in range(first_non_empty_idx):
        vol_org.append(empty_slice)
        vol_linear.append(empty_slice)
        vol_sam2.append(empty_slice)

        vol_org.append(empty_slice)
        vol_linear.append(empty_slice)
        vol_sam2.append(empty_slice)

    for idx in range(first_non_empty_idx, last_non_empty_idx+1):
        base_name = os.path.basename(org_files[idx]).replace(".png", "")
        img_org = cv2.imread(org_files[idx], cv2.IMREAD_GRAYSCALE)

        if Quantative:
            vol_org.append(empty_slice)
            vol_linear.append(empty_slice)
            vol_sam2.append(empty_slice)
        else:
            # Baseline: just original slices
            vol_org.append(img_org)
            # Linear and SAM2 alternating slices
            vol_linear.append(img_org)
            vol_sam2.append(img_org)
        
        if idx >= first_non_empty_idx and idx < last_non_empty_idx-1:
            interp_name = base_name + "_5"
            sam_name = base_name + "_5S"
            if interp_name in lin_interp_files: 
                img_lin = cv2.imread(lin_interp_files[interp_name], cv2.IMREAD_GRAYSCALE)
            else:
                img_lin = np.zeros_like(img_org)
            
            if sam_name in sam2_dict:
                img_sam = cv2.imread(sam2_dict[sam_name], cv2.IMREAD_GRAYSCALE)
            else:
                img_sam = np.zeros_like(img_org)
            
            vol_org.append(img_org)
            vol_linear.append(img_lin)
            vol_sam2.append(img_sam)

    while len(vol_org) < (len(org_files)*2)-2:
        vol_org.append(empty_slice)
        vol_linear.append(empty_slice)
        vol_sam2.append(empty_slice)
        vol_org.append(empty_slice)
        vol_linear.append(empty_slice)
        vol_sam2.append(empty_slice)

    vol_org = np.stack(vol_org, axis=0)
    vol_linear = np.stack(vol_linear, axis=0)
    vol_sam2 = np.stack(vol_sam2, axis=0)

    vol_linear = (vol_linear > 0)  # ensure binary
    vol_linear = vol_linear.astype(np.uint8) * 255

    return vol_org, vol_linear, vol_sam2

# -----------------------------
# Extract original NIfTI views
# -----------------------------
def extract_original_views(case_dir, out_dir, i, xc, yc, px, py, pz):
    os.makedirs(out_dir, exist_ok=True)
    t2_path = os.path.join(case_dir, "t2.nii.gz")
    seg_path = os.path.join(case_dir, "t2_anatomy_reader1.nii.gz")

    img = nib.load(t2_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()

    # CRITICAL: convert to (Z,Y,X)
    img = np.transpose(img, (2, 1, 0))
    seg = np.transpose(seg, (2, 1, 0))
    Z, Y, X = img.shape

    xc = np.clip(xc, 0, X - 1)
    yc = np.clip(yc, 0, Y - 1)

    # -------- Sagittal --------
    sag_img = img[:, :, xc]
    sag_seg = seg[:, :, xc]
    sag_img = np.flip(sag_img, axis=0)
    sag_seg = np.flip(sag_seg, axis=0)
    new_height = int(round(sag_img.shape[0] * pz / py))
    sag_img = cv2.resize(sag_img, (sag_img.shape[1], new_height), interpolation=cv2.INTER_LINEAR)
    sag_seg = cv2.resize(sag_seg, (sag_seg.shape[1], new_height), interpolation=cv2.INTER_NEAREST)

    # -------- Coronal --------
    cor_img = img[:, yc, :]
    cor_seg = seg[:, yc, :]
    cor_img = np.flip(cor_img, axis=0)
    cor_seg = np.flip(cor_seg, axis=0)
    new_height = int(round(cor_img.shape[0] * pz / px))
    cor_img = cv2.resize(cor_img, (cor_img.shape[1], new_height), interpolation=cv2.INTER_LINEAR)
    cor_seg = cv2.resize(cor_seg, (cor_seg.shape[1], new_height), interpolation=cv2.INTER_NEAREST)

    # Normalize
    sag_img = ((sag_img - sag_img.min()) / (sag_img.max() - sag_img.min() + 1e-8) * 255).astype(np.uint8)
    cor_img = ((cor_img - cor_img.min()) / (cor_img.max() - cor_img.min() + 1e-8) * 255).astype(np.uint8)
    sag_seg = (sag_seg > 0).astype(np.uint8) * 255
    cor_seg = (cor_seg > 0).astype(np.uint8) * 255

    # Save
    cv2.imwrite(os.path.join(out_dir, f"orig_img_sagittal_{i:03d}.png"), sag_img)
    cv2.imwrite(os.path.join(out_dir, f"orig_seg_sagittal_{i:03d}.png"), sag_seg)
    cv2.imwrite(os.path.join(out_dir, f"orig_img_coronal_{i:03d}.png"), cor_img)
    cv2.imwrite(os.path.join(out_dir, f"orig_seg_coronal_{i:03d}.png"), cor_seg)

    z_target = sag_img.shape[0]

    return z_target

# ===============================
# MAIN LOOP
# ===============================
test_train = "test"
cases = list(range(1, 20))

for i in cases:
    case_dir = fr"C:\Users\20213282\Documents\uni\BEP\codes\data\data_prostate\{test_train}\{i:03d}"
    mask_dir = os.path.join(case_dir, "masks")
    sam2mask_dir = os.path.join(case_dir, "sam2_masks")
    out_dir = os.path.join(case_dir, "views")
    os.makedirs(out_dir, exist_ok=True)

    px, py, pz = get_voxel_spacing(case_dir)

    seg_path = os.path.join(case_dir, "t2_anatomy_reader1.nii.gz")
    seg = nib.load(seg_path).get_fdata()
    target_z = seg.shape[2]  # original axial slice count

    # Compute center
    zc, yc, xc = compute_center_pixel(seg)
    print(f"Center coords normal: x={xc}, y={yc}, z={zc}")

    coord_file = os.path.join(out_dir, f"center_coordinates_{i:03d}.txt")
    with open(coord_file, "w") as f:
        f.write(f"{xc},{yc},{zc}")


    # Extract views
    target_z = seg.shape[2]  # original axial slice count
    target_y = seg.shape[1]  # original Y dimension
    target_x = seg.shape[0]  # original X dimension (after transpose)

    # Build volumes
    vol_org, vol_linear, vol_sam2 = build_spaced_volumes(mask_dir, sam2mask_dir, px, pz,Quantative=False)
    vol_org_quan, vol_linear_quan, vol_sam2_quan = build_spaced_volumes(mask_dir, sam2mask_dir, px, pz, Quantative=True)
    # Extract views
    num_original_slices = seg.shape[2]
    target_z = extract_original_views(case_dir, out_dir, i, xc, yc, px, py, pz)

    extract_center_views(vol_org, out_dir, "Baseline", xc, yc, i, px, py, pz,target_z)
    extract_center_views(vol_linear, out_dir, "linear", xc, yc, i, px, py, pz,target_z)
    extract_center_views(vol_sam2, out_dir, "sam2", xc, yc, i, px, py, pz,target_z)

    extract_center_views(vol_org_quan, out_dir, "Baseline_Quan", xc, yc, i, px, py, pz,target_z)
    extract_center_views(vol_linear_quan, out_dir, "linear_Quan", xc, yc, i, px, py, pz,target_z)
    extract_center_views(vol_sam2_quan, out_dir, "sam2_Quan", xc, yc, i, px, py, pz,target_z)

  