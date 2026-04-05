import os
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion
import nibabel as nib

# =========================
# HELPER FUNCTIONS
# =========================

def get_voxel_spacing(case_dir):
    """
    Load a NIfTI file (t2.nii.gz) and return voxel spacing in mm.
    """
    t2_path = os.path.join(case_dir, "t2.nii.gz")
    if not os.path.exists(t2_path):
        raise FileNotFoundError(f"Missing t2.nii.gz in {case_dir}")
    nii = nib.load(t2_path)
    return nii.header.get_zooms()[:3]  # px, py, pz

def get_surface(mask):
    """
    Compute the surface voxels of a binary mask using erosion.
    The surface is defined as voxels present in mask but not in eroded mask.
    """
    eroded = binary_erosion(mask)
    return mask ^ eroded    # XOR to get surface voxels
# =========================
# METRICS
# =========================

def dice_score(mask1, mask2):
    """
    Compute the Dice similarity coefficient between two binary masks.
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    """
    m1 = (mask1 > 0).astype(np.uint8)
    m2 = (mask2 > 0).astype(np.uint8)
    intersection = np.sum(m1 * m2)
    total = np.sum(m1) + np.sum(m2)
    if total == 0:   # Both masks empty -> perfect match
        return 1.0
    return 2.0 * intersection / total

def hausdorff_95(mask1, mask2, spacing):
    """
    Compute the 95th percentile of the Hausdorff distance between two masks.
    - Extract surface voxels
    - Scale by voxel spacing to convert to mm
    - Compute distances using cKDTree for efficiency
    """
    pts1 = np.argwhere(get_surface(mask1))
    pts2 = np.argwhere(get_surface(mask2))
    if len(pts1) == 0 or len(pts2) == 0:
        return np.nan
    
    # Convert voxel coordinates to real-world spacing (mm)
    pts1 = pts1 * spacing
    pts2 = pts2 * spacing

    # Build KD-Trees for fast nearest neighbor queries
    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)
    
    dists1, _ = tree2.query(pts1)
    dists2, _ = tree1.query(pts2)

    # 95th percentile of distances
    hd95 = max(np.percentile(dists1, 95), np.percentile(dists2, 95))
    return hd95

def Average_Symetrical_surface_distance(mask1, mask2, spacing):
    """
    Compute the Average Symmetrical Surface Distance (ASSD) between two masks.
    Similar to Hausdorff but takes the mean of all nearest surface distances.
    """
    pts1 = np.argwhere(get_surface(mask1)) * spacing
    pts2 = np.argwhere(get_surface(mask2)) * spacing
    if len(pts1) == 0 or len(pts2) == 0:
        return np.nan
    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)
    dists1, _ = tree2.query(pts1)
    dists2, _ = tree1.query(pts2)
    return (np.mean(dists1) + np.mean(dists2)) / 2

# =========================
# MAIN EVALUATION
# =========================

def evaluate_case(view_dir, case):
    """
    Evaluate all views (sagittal, coronal) of a single case.
    Compute Dice, Hausdorff, and ASSD metrics between:
    - Linear interpolation (LI)
    - SAM output
    - Baseline
    - Original Nifti segmentation
    """
    results = []

    views = ["sagittal", "coronal"]

    case_dir_2 = os.path.join(base_dir, f"{case:03d}")
    px, py, pz = get_voxel_spacing(case_dir_2)

    for view in views:
        # Build paths to segmentation images, all these masks besides base_2 have original slices removed
        # This means that for base only the copies of the originals remain
        lin_path = os.path.join(view_dir, f"linear_Quan_{view}_{case:03d}.png")
        sam_path = os.path.join(view_dir, f"sam2_Quan_{view}_{case:03d}.png")
        base_path = os.path.join(view_dir, f"Baseline_Quan_{view}_{case:03d}.png")
        base_2_path = os.path.join(view_dir, f"Baseline_{view}_{case:03d}.png")
        orig_path = os.path.join(view_dir, f"orig_seg_{view}_{case:03d}.png")

        # Skip evaluation if any file missing
        if not all(os.path.exists(p) for p in [lin_path, sam_path, base_path, orig_path]):
            continue

        # Load images in grayscale
        lin = cv2.imread(lin_path, cv2.IMREAD_GRAYSCALE)
        sam = cv2.imread(sam_path, cv2.IMREAD_GRAYSCALE)
        base = cv2.imread(base_path, cv2.IMREAD_GRAYSCALE)
        orig = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
        base_2= cv2.imread(base_2_path, cv2.IMREAD_GRAYSCALE)   #

        # --- Compute Dice scores ---
        dice_base_lin = dice_score(base, lin)
        dice_base_sam = dice_score(base, sam)
        dice_lin_sam = dice_score(lin, sam)
        dice_base_orig = dice_score(base_2, orig)

        # Set voxel spacing according to view plane
        if view == "coronal":
            spacing = np.array([px, pz])
        elif view == "sagittal":
            spacing = np.array([py, pz])

        # --- Compute Hausdorff 95 distances ---
        haus_base_lin = hausdorff_95(base, lin, spacing)
        haus_base_sam = hausdorff_95(base, sam, spacing)
        haus_base_orig = hausdorff_95(base_2, orig, spacing)
        haus_lin_sam = hausdorff_95(lin, sam, spacing)

        # --- Compute Average Symmetrical Surface Distances ---
        assd_base_lin = Average_Symetrical_surface_distance(base, lin, spacing)
        assd_base_sam = Average_Symetrical_surface_distance(base, sam, spacing)
        assd_base_orig = Average_Symetrical_surface_distance(base_2, orig, spacing)
        assd_lin_sam = Average_Symetrical_surface_distance(lin, sam, spacing)

        # Append results for this view
        results.append({
            "case": case,
            "view": view,
            # Linear interpolation vs SAM results (only real interesting results)
            "dice LI vs SAM": dice_lin_sam,
            "HF95 LI vs SAM": haus_lin_sam,
            "AHD LI vs SAM": assd_lin_sam,
            "Debug_results": "Remaining result",

            # Remaining Dice Scores
            "dice Base vs LI": dice_base_lin,
            "dice Base vs SAM": dice_base_sam,
            
            # Remaining Hausdorff 95%
            "HF95 Base vs LI": haus_base_lin,
            "HF95 Base vs SAM": haus_base_sam,

            # Remaining ASSD
            "assd Base vs LI": assd_base_lin,
            "assd Base vs SAM": assd_base_sam,
            "assd Base vs Orig": assd_base_orig,

            # Compare the full baseline to the Nifti original segmentation
            # for troubleshooting
            "Nifti vs Base comparison":"Niftvsbase",
            "dice Base vs Orig": dice_base_orig,
            "HF95 base vs orig": haus_base_orig,
            "assd Bas vs Orig":assd_base_orig
            
        })

    return results

# =========================
# LOOP OVER DATASET
# =========================

train_test = "test"
cases = list(range(1, 20)) # Case IDs from 1 to 19 for test data
#train_test = "train"
#cases = list(range(20, 159)) # Case IDs from 20 to 158 for training data


base_dir = fr"C:\Users\20213282\Documents\uni\BEP\codes\data\data_prostate\{train_test}"
output_dir = fr"C:\Users\20213282\Documents\uni\BEP\codes\data\data_prostate"

all_results = []

for case in cases:
    case_dir = os.path.join(base_dir, f"{case:03d}", "views")
    if not os.path.exists(case_dir):
        continue
    case_results = evaluate_case(case_dir, case)
    all_results.extend(case_results)
    print(f"Processed case {case:03d}")

# =========================
# SAVE RESULTS TO EXCEL
# =========================

df = pd.DataFrame(all_results)

# Sort by view (coronal first) and case number
view_order = {"coronal": 0, "sagittal": 1}
df["view_order"] = df["view"].map(view_order)
df = df.sort_values(by=["view_order", "case"]).drop(columns=["view_order"])

output_path = os.path.join(output_dir, f"evaluation_results_{train_test}.xlsx")
df.to_excel(output_path, index=False)
print(f"\nSaved results to: {output_path}")