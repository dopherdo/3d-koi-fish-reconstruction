import cv2
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from camutils import triangulate

def decode(imprefix,start,threshold):
    """
    Given a sequence of 20 images of a scene showing projected 10 bit gray code, 
    decode the binary sequence into a decimal value in (0,1023) for each pixel.
    Mark those pixels whose code is likely to be incorrect based on the user 
    provided threshold.  Images are assumed to be named "imageprefixN.png" where
    N is a 2 digit index (e.g., "img00.png,img01.png,img02.png...")
 
    Parameters
    ----------
    imprefix : str
       Image name prefix
      
    start : int
       Starting index
       
    threshold : float
       Threshold to determine if a bit is decodeable
       
    Returns
    -------
    code : 2D numpy.array (dtype=float)
        Array the same size as input images with entries in (0..1023)
        
    mask : 2D numpy.array (dtype=logical)
        Array indicating which pixels were correctly decoded based on the threshold
    
    """
    
    # we will assume a 10 bit code
    nbits = 10
    image_shape = None
    gray_bits = []

    bad_mask = None  # will become OR of all bad pixels across bit pairs

    for i in range(nbits):
        # construct filenames
        idx1 = start + 2*i
        idx2 = start + 2*i + 1
        fname1 = f"{imprefix}{idx1:02d}_u.png"
        fname2 = f"{imprefix}{idx2:02d}_u.png"
        
        # read and normalize
        img1 = plt.imread(fname1).astype(np.float32)
        img2 = plt.imread(fname2).astype(np.float32)
        # Debug prints for image loading
        print(f"[DEBUG] Loading {fname1}: shape={img1.shape}, dtype={img1.dtype}, min={img1.min()}, max={img1.max()}")
        print(f"[DEBUG] Loading {fname2}: shape={img2.shape}, dtype={img2.dtype}, min={img2.min()}, max={img2.max()}")
        if img1.ndim == 3:
            img1 = img1.mean(axis=2)
            img2 = img2.mean(axis=2)
        
        if image_shape is None:
            image_shape = img1.shape
        
        # gray bit = 1 where img1 > img2
        bit = (img1 > img2).astype(np.uint8)
        gray_bits.append(bit)

        # undecodable if abs diff < threshold
        bad = (np.abs(img1 - img2) < threshold)
        bad_mask = bad if bad_mask is None else (bad_mask | bad)

    # Stack bits into 3D array (shape: [nbits, H, W])
    gray_stack = np.stack(gray_bits, axis=0)

    # Convert from gray code to binary code
    binary_stack = np.zeros_like(gray_stack)
    binary_stack[0] = gray_stack[0]
    for i in range(1, nbits):
        binary_stack[i] = binary_stack[i-1] ^ gray_stack[i]  # XOR successive bits

    # Compute final decimal code
    powers = 2**np.arange(nbits-1, -1, -1).reshape((nbits, 1, 1))  # MSB to LSB
    code = np.sum(binary_stack * powers, axis=0).astype(np.float32)

    # Final mask: valid where all bits were decodable
    mask = ~bad_mask

        
    return code,mask

def compute_foreground_mask(bg_img, obj_img, threshold=30, visualize=False):
    diff = np.abs(obj_img - bg_img)
    diff_gray = cv2.cvtColor((diff * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
    # Morphological cleaning
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Keep only the largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8) * 255
    if visualize:
        cv2.imshow("Foreground Mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return mask.astype(bool)

def load_color_images(base_path, cam="C0"):
    bg_path = os.path.join(base_path, f"color_{cam}_00_u.png")
    obj_path = os.path.join(base_path, f"color_{cam}_01_u.png")
    bg = cv2.imread(bg_path).astype(np.float32) / 255.0
    obj = cv2.imread(obj_path).astype(np.float32) / 255.0
    # Resize both to 1080p to match valid_mask shape
    target_size = (1920, 1080)
    bg = cv2.resize(bg, target_size, interpolation=cv2.INTER_AREA)
    obj = cv2.resize(obj, target_size, interpolation=cv2.INTER_AREA)
    return bg, obj

def apply_combined_mask(valid_mask, fg_mask):
    return np.logical_and(valid_mask, fg_mask)

def extract_colors(obj_img0, obj_img1, mask):
    colors0 = obj_img0[mask]
    colors1 = obj_img1[mask]
    blended = (colors0 + colors1) / 2.0
    return blended.T

def validate_points(pts3d, P0, P1, uv0, uv1, mask, max_reproj_error=2.0):
    valid = np.ones(pts3d.shape[1], dtype=bool)
    for i in range(pts3d.shape[1]):
        if not mask[i]:
            valid[i] = False
            continue
        pt = np.append(pts3d[:, i], 1)
        if (P0[2, :] @ pt) <= 0 or (P1[2, :] @ pt) <= 0:
            valid[i] = False
            continue
        proj0 = P0 @ pt
        proj1 = P1 @ pt
        proj0 = proj0[:2] / proj0[2]
        proj1 = proj1[:2] / proj1[2]
        error0 = np.linalg.norm(proj0 - uv0[:, i])
        error1 = np.linalg.norm(proj1 - uv1[:, i])
        if error0 > max_reproj_error or error1 > max_reproj_error:
            valid[i] = False
    return valid

def reconstruct(imprefixL, imprefixR, threshold, camL, camR, colorL_bg_path, colorL_obj_path, colorR_bg_path, colorR_obj_path, mask_threshold=30, visualize_mask=False):
    """
    Full reconstruct function per assignment prompt:
    - Loads color images for foreground masking
    - Loads pattern images for decoding
    - Combines the foreground and decoding masks
    - Triangulates only valid foreground+decoded points
    - Extracts color for each triangulated point
    - Includes debug prints for each step
    - Handles empty results gracefully
    """
    # 1. Decode the H and V coordinates for the two views (pattern images)
    HL, HmaskL = decode(imprefixL, 0, threshold)
    VL, VmaskL = decode(imprefixL, 20, threshold)
    HR, HmaskR = decode(imprefixR, 0, threshold)
    VR, VmaskR = decode(imprefixR, 20, threshold)

    # 2. Construct the combined 20 bit code and mask for each view
    CL = HL + 1024 * VL
    CR = HR + 1024 * VR
    maskL_decode = HmaskL & VmaskL
    maskR_decode = HmaskR & VmaskR

    # 3. Foreground Masking using color images
    colorL_bg = cv2.imread(colorL_bg_path).astype(np.float32) / 255.0
    colorL_obj = cv2.imread(colorL_obj_path).astype(np.float32) / 255.0
    colorR_bg = cv2.imread(colorR_bg_path).astype(np.float32) / 255.0
    colorR_obj = cv2.imread(colorR_obj_path).astype(np.float32) / 255.0

    diffL = np.abs(colorL_obj - colorL_bg)
    diffL_gray = cv2.cvtColor((diffL * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    _, fg_maskL = cv2.threshold(diffL_gray, mask_threshold, 255, cv2.THRESH_BINARY)
    fg_maskL = fg_maskL.astype(bool)

    diffR = np.abs(colorR_obj - colorR_bg)
    diffR_gray = cv2.cvtColor((diffR * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    _, fg_maskR = cv2.threshold(diffR_gray, mask_threshold, 255, cv2.THRESH_BINARY)
    fg_maskR = fg_maskR.astype(bool)

    if visualize_mask:
        cv2.imshow("Foreground Mask L", fg_maskL.astype(np.uint8)*255)
        cv2.imshow("Foreground Mask R", fg_maskR.astype(np.uint8)*255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Debug prints for mask statistics
    print(f"[DEBUG] Decoding maskL valid pixels: {np.sum(maskL_decode)}")
    print(f"[DEBUG] Decoding maskR valid pixels: {np.sum(maskR_decode)}")
    print(f"[DEBUG] Foreground maskL valid pixels: {np.sum(fg_maskL)}")
    print(f"[DEBUG] Foreground maskR valid pixels: {np.sum(fg_maskR)}")

    # 4. Combine with decoding masks
    maskL = maskL_decode & fg_maskL
    maskR = maskR_decode & fg_maskR
    print(f"[DEBUG] Combined maskL valid pixels: {np.sum(maskL)}")
    print(f"[DEBUG] Combined maskR valid pixels: {np.sum(maskR)}")

    # 5. Matching codes as before
    h, w = CL.shape
    CL_flat = CL.flatten()
    CR_flat = CR.flatten()
    maskL_flat = maskL.flatten()
    maskR_flat = maskR.flatten()

    CL_valid = CL_flat[maskL_flat]
    CR_valid = CR_flat[maskR_flat]

    matches, matchL, matchR = np.intersect1d(CL_valid, CR_valid, return_indices=True)
    print(f"[DEBUG] Number of matched correspondences: {len(matchL)}")

    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    xx = xx.flatten()
    yy = yy.flatten()

    validL_x = xx[maskL_flat]
    validL_y = yy[maskL_flat]
    validR_x = xx[maskR_flat]
    validR_y = yy[maskR_flat]

    if len(matchL) == 0 or len(matchR) == 0:
        print("[WARNING] No valid correspondences for triangulation.")
        return np.zeros((2,0)), np.zeros((2,0)), np.zeros((3,0)), np.zeros((3,0))

    pts2L = np.stack((validL_x[matchL], validL_y[matchL]), axis=0)
    pts2R = np.stack((validR_x[matchR], validR_y[matchR]), axis=0)

    # 6. Extract color for each triangulated point
    colorsL = colorL_obj[validL_y[matchL], validL_x[matchL]].T  # shape (3, N)
    colorsR = colorR_obj[validR_y[matchR], validR_x[matchR]].T
    colors = (colorsL + colorsR) / 2.0

    # 7. Triangulate the points
    pts3 = triangulate(pts2L, camL, pts2R, camR)

    return pts2L, pts2R, pts3, colors
