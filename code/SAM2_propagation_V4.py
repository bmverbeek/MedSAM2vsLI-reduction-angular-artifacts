import os
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import sys

# Add SAM2 code directory to Python path
sys.path.insert(0, r"C:\Users\20213282\Documents\uni\BEP\codes\SAM_2")

# =========================
# HELPER FUNCTIONS
# =========================

def is_original(fname):
    """
    Check if a frame is an original slice (not interpolated or propagated).
    Original frames do NOT contain '_5' in filename.
    """
    return "_5" not in fname

def get_base_index(fname):
    """
    Extract the base index from filename (first number before underscores).
    Example: '00001_5.png' -> 1
    """
    return int(os.path.splitext(fname)[0].split('_')[0])

def sort_key(fname):
    """
    Generate a sorting key for filenames to ensure proper frame order.
    Handles names like:
      - '00001.jpg' -> (1,0,0)
      - '00001_5_1.png' -> (1,5,1)
    """
    parts = os.path.splitext(fname)[0].split('_')
    base = int(parts[0])
    if len(parts) == 1:
        return (base, 0, 0)
    elif len(parts) == 3:
        return (base, int(parts[1]), int(parts[2]))
    return (base, 0, 0)

def load_mask(path):
    """
    Load a mask from a PNG file.
    Returns a boolean array where True indicates foreground.
    """
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(bool)

def propagate_direction(predictor, state, anchor_idx, frame_names, direction='forward'):
    """
    Propagate the SAM2 mask in one direction (forward/backward) until a valid mask is found.
    - predictor: SAM2 video predictor
    - state: current propagation state
    - anchor_idx: index of anchor/original frame
    - frame_names: list of frame filenames
    - direction: 'forward' or 'backward'
    
    Returns the first non-empty propagated mask as uint8 0-255, or None if not found.
    """
    reverse = direction == 'backward'
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state, reverse=reverse):
        # skip frames before/after anchor
        if (reverse and frame_idx >= anchor_idx) or (not reverse and frame_idx <= anchor_idx):
            continue
        # stop if we reach another original
        if is_original(frame_names[frame_idx]):
            break
        mask_arr = (mask_logits[0] > 0).detach().cpu().numpy()
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[0]
        return mask_arr.astype(np.uint8) * 255
    return None

# =========================
# MAIN FUNCTION
# =========================

def propagate_sam2_prostate(case_id: str,
                             img_root_dir: str,
                             mask_root_dir: str,
                             output_dir: str,
                             sam2_checkpoint: str,
                             model_cfg: str,
                             ann_obj_id: int = 1):
    """
    Propagate prostate segmentation masks using SAM2 across video slices.
    
    Steps:
    1. Initialize SAM2 video predictor
    2. Iterate over original frames (not interpolated)
    3. For each anchor frame:
        - Load its mask
        - Initialize SAM2 state
        - Add anchor mask to predictor
        - Propagate forward and backward
        - Store valid propagated masks
    4. Merge forward/backward masks and save (_5S files)
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load and sort frame filenames
    frame_names = sorted([f for f in os.listdir(img_root_dir) if f.lower().endswith((".jpg", ".jpeg"))], key=sort_key)
    print(f"Total frames: {len(frame_names)}")

    # Initialize SAM2 predictor once
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    predictor.to(device)
    
    video_segments = {}  # Dictionary: anchor_id -> {'forward': mask, 'backward': mask}

    # Iterate over anchor/original frames
    for anchor_idx, anchor_name in enumerate(frame_names):
        if not is_original(anchor_name):
            continue

        mask_path = os.path.join(mask_root_dir, anchor_name.replace(".jpg", ".png"))
        if not os.path.exists(mask_path):
            continue

        mask = load_mask(mask_path)
        if mask.sum() == 0:
            continue  # skip empty anchors

        # Initialize SAM2 state for current anchor  
        state = predictor.init_state(img_root_dir)
        predictor.add_new_mask(state, frame_idx=anchor_idx, obj_id=ann_obj_id, mask=mask)

        anchor_id = get_base_index(anchor_name)

        # Propagate in both directions
        for dir_name in ['forward', 'backward']:
            result_mask = propagate_direction(predictor, state, anchor_idx, frame_names, dir_name)
            # --- Only keep non-empty masks ---
            if result_mask is not None and np.any(result_mask > 0):
                if anchor_id not in video_segments:
                    video_segments[anchor_id] = {}
                video_segments[anchor_id][dir_name] = result_mask

    # =========================
    # MERGE + SAVE (_5S ONLY)
    # =========================

    for anchor_id, masks in video_segments.items():
        # If either direction missing, save empty mask
        if "forward" not in masks or "backward" not in masks:
            empty_mask = np.zeros_like(load_mask(os.path.join(mask_root_dir, f"{anchor_id:05d}.png")), dtype=np.uint8)
            out_name = f"{anchor_id:05d}_5S.png"
            Image.fromarray(empty_mask).save(os.path.join(output_dir, out_name))
            continue

        forward = masks["forward"].astype(np.float32) / 255.0
        backward = masks["backward"].astype(np.float32) / 255.0

        # Merge forward/backward masks with linear interpolation
        merged = 0.5 * forward + 0.5 * backward

         # Binarize merged mask, creating a mask union
        merged = (merged >= 0.5).astype(np.uint8) * 255

        out_name = f"{anchor_id:05d}_5S.png"
        out_path = os.path.join(output_dir, out_name)
        Image.fromarray(merged).save(out_path)

# =========================
# Main Loop
# =========================

test_train = "test"
cases = list(range(1,20))
# Alternative for training data:
# test_train = "train"
# cases = list(range(20,159))


for i in cases: 
    img_dir = fr"C:\Users\20213282\Documents\uni\BEP\codes\data\data_prostate\{test_train}\{i:03d}\images"
    mask_dir = fr"C:\Users\20213282\Documents\uni\BEP\codes\data\data_prostate\{test_train}\{i:03d}\masks"
    output_dir = fr"C:\Users\20213282\Documents\uni\BEP\codes\data\data_prostate\{test_train}\{i:03d}\sam2_masks"

    sam2_checkpoint = r"C:\Users\20213282\Documents\uni\BEP\codes\checkpoints\MEDSAM2_latest.pt"
    model_cfg = r"C:\Users\20213282\Documents\uni\BEP\codes\checkpoints\sam2.1_hiera_t512.yaml"

    propagate_sam2_prostate(i, img_dir, mask_dir, output_dir, sam2_checkpoint, model_cfg)