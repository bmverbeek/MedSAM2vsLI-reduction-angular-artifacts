import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# =========================
# HELPER FUNCTIONS
# =========================

def load_image(path):
    """
    Load an image from disk and return as NumPy array.
    Used for background MRI images.
    """
    return np.array(Image.open(path))

def load_mask(path, shape):
    """
    Load a mask image and return a boolean array.
    
    - If file does not exist → return empty mask
    - Converts RGB masks to grayscale if needed
    - Resizes mask to match given shape
    """
    if not os.path.exists(path):
        return np.zeros(shape, dtype=bool)
    m = np.array(Image.open(path))
    # Convert RGB mask to single channel if needed
    if m.ndim == 3:
        m = m[..., 0]

    # Resize if shape mismatch
    if m.shape != shape:
        m = np.array(Image.fromarray(m).resize((shape[1], shape[0]), resample=Image.NEAREST)) # preserve mask values
    return (m > 0) # convert to boolean mask

def get_contour(mask, thickness=1):
    """
    Extract contour (edges) of a binary mask.
    Done via iterative erosion using neighborhood shifts.
    
    thickness controls how thick the contour is.
    """
    eroded = mask.copy()
    for _ in range(thickness):
        # Shrink mask using logical AND with shifted versions
        eroded &= np.roll(eroded, 1, 0)
        eroded &= np.roll(eroded, -1, 0)
        eroded &= np.roll(eroded, 1, 1)
        eroded &= np.roll(eroded, -1, 1)
    return mask & (~eroded)     # contour = original - eroded

# =========================
# OVERLAY FUNCTION
# =========================

def overlay_image(img, masks, colors, alphas, contours=None, show_overlap=False, overlap_color=[0.6,0,0.8],alpha_overlap=[0.5]):
    """
    Overlay multiple masks on top of a grayscale image.

    Parameters:
    - img: background image
    - masks: list of boolean masks
    - colors: list of RGB colors (0-1)
    - alphas: transparency values
    - contours: list indicating whether to show contour only
    - show_overlap: highlight overlap between first two masks
    """
    H, W = img.shape[:2]
    fig, ax = plt.subplots()

    # Show base image
    ax.imshow(img, cmap="gray")

    # -------------------------
    # Compute overlap (optional)
    # -------------------------
    if show_overlap and len(masks) >= 2:
        overlap_mask = masks[0] & masks[1]

        # Remove overlap from individual masks
        masks[0] = masks[0] & (~overlap_mask)
        masks[1] = masks[1] & (~overlap_mask)

        # Create overlay for overlap
        overlay = np.zeros((H, W, 4))
        overlay[overlap_mask] = overlap_color + alpha_overlap
        ax.imshow(overlay)

    # -------------------------
    # Overlay individual masks
    # -------------------------
    for i, mask in enumerate(masks):
        overlay = np.zeros((H, W, 4))
        
        # If contour mode → only draw edges
        if contours and contours[i]:
            mask = get_contour(mask)

        overlay[mask] = list(colors[i]) + [alphas[i]]
        ax.imshow(overlay)

    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig

# =========================
# VIEWER FUNCTION
# =========================

def run_viewer(base_dir, out_dir, case, train_test="test", show=True):
    """
    Load images and segmentation results for one case,
    overlay them, and save visualization images.

    Combines:
    - Original MRI
    - Linear interpolation masks
    - SAM2 masks
    - Baseline masks
    - NIfTI ground truth
    """
    view_dir = os.path.join(base_dir, "views")
    os.makedirs(out_dir, exist_ok=True)

    # Define filenames
    files = {
        "orig_cor": f"orig_img_coronal_{case:03d}.png",
        "orig_sag": f"orig_img_sagittal_{case:03d}.png",
        "lin_cor": f"linear_coronal_{case:03d}.png",
        "lin_sag": f"linear_sagittal_{case:03d}.png",
        "sam_cor": f"sam2_coronal_{case:03d}.png",
        "sam_sag": f"sam2_sagittal_{case:03d}.png",
        "baseline_cor": f"baseline_coronal_{case:03d}.png",
        "baseline_sag": f"baseline_sagittal_{case:03d}.png",
        "nifti_cor": f"orig_set_coronal_{case:03d}.png",
        "nifti_sag": f"orig_seg_sagittal_{case:03d}.png",
    }

    # -------------------------
    # Load base images
    # -------------------------
    orig_cor = load_image(os.path.join(view_dir, files["orig_cor"]))
    orig_sag = load_image(os.path.join(view_dir, files["orig_sag"]))

    Hc, Wc = orig_cor.shape[:2]
    Hs, Ws = orig_sag.shape[:2]

    # -------------------------
    # Load masks
    # -------------------------
    lin_cor = load_mask(os.path.join(view_dir, files["lin_cor"]), (Hc, Wc))
    lin_sag = load_mask(os.path.join(view_dir, files["lin_sag"]), (Hs, Ws))

    sam_cor = load_mask(os.path.join(view_dir, files["sam_cor"]), (Hc, Wc))
    sam_sag = load_mask(os.path.join(view_dir, files["sam_sag"]), (Hs, Ws))

    baseline_cor = load_mask(os.path.join(view_dir, files["baseline_cor"]), (Hc, Wc))
    baseline_sag = load_mask(os.path.join(view_dir, files["baseline_sag"]), (Hs, Ws))

    nifti_cor = load_mask(os.path.join(view_dir, files["nifti_cor"]), (Hc, Wc))
    nifti_sag = load_mask(os.path.join(view_dir, files["nifti_sag"]), (Hs, Ws))

    # -------------------------
    # Visualization settings
    # -------------------------
    colors_linear = [0.6, 0, 0.8]       # purple
    colors_sam =    [1.0, 0, 0]         # red
    colors_baseline=[1.0, 0.9, 0.1]     # yellow
    colors_nifti =  [0, 1, 1]           # cyan
    overlap_color = [0,1,0]             # green

    alpha = [0.6, 0.6, 1.0,0.5]
    alpha_overlap = [0.19]

    # Define views (coronal & sagittal)
    views = {
        "coronal": (orig_cor, [lin_cor, sam_cor, baseline_cor, nifti_cor], [colors_linear, colors_sam, colors_baseline, colors_nifti], alpha, [False, False, True, True]),
        "sagittal": (orig_sag, [lin_sag, sam_sag, baseline_sag, nifti_sag], [colors_linear, colors_sam, colors_baseline, colors_nifti], alpha, [False, False, True, True])
    }

    # -------------------------
    # Generate and save overlays
    # -------------------------
    for view, (img, masks, colors, alphas, contours) in views.items():
        fig = overlay_image(img, masks, colors, alphas, contours, show_overlap=True, overlap_color=overlap_color,alpha_overlap=alpha_overlap)

        save_path = os.path.join(out_dir, f"{case:03d}_{view}.png")

        fig.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=300)

        if show:
            plt.show()
        plt.close(fig)

# =========================
# Main loop
# =========================

train_test = "test"
cases = list(range(1, 20))
out_dir = r"C:\Users\20213282\Pictures\result_images"
for case in cases:
    base_dir = fr"C:\Users\20213282\Documents\uni\BEP\codes\data\data_prostate\{train_test}\{case:03d}"
    run_viewer(base_dir, out_dir, case, show=False)
