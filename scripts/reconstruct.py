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

def reconstruct(imprefixL, imprefixR, threshold=0.005, camL=None, camR=None, colorL_bg_path=None, colorL_obj_path=None, colorR_bg_path=None, colorR_obj_path=None, mask_threshold=20, visualize_mask=True):
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
    # --- Tuning parameters ---
    dilation_kernel_size = (9, 9)  # Larger kernel to fill more holes
    dilation_iterations = 2         # More aggressive dilation
    mask_threshold = mask_threshold # Lowered for more inclusion
    decoding_threshold = threshold  # Lowered for more inclusion
    outlier_stddev = 2.5            # 3D outlier removal stddev
    # -------------------------

    # 1. Decode the H and V coordinates for the two views (pattern images)
    HL, HmaskL = decode(imprefixL, 0, decoding_threshold)
    VL, VmaskL = decode(imprefixL, 20, decoding_threshold)
    HR, HmaskR = decode(imprefixR, 0, decoding_threshold)
    VR, VmaskR = decode(imprefixR, 20, decoding_threshold)

    # 2. Construct the combined 20 bit code and mask for each view
    CL = HL + 1024 * VL
    CR = HR + 1024 * VR
    maskL_decode = HmaskL & VmaskL
    maskR_decode = HmaskR & VmaskR

    # Visualize decoding masks
    if visualize_mask:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.title('Decoding Mask L')
        plt.imshow(maskL_decode, cmap='gray')
        plt.subplot(1,2,2)
        plt.title('Decoding Mask R')
        plt.imshow(maskR_decode, cmap='gray')
        plt.show()

    # 3. Foreground Masking using color images (refined)
    colorL_bg = cv2.imread(colorL_bg_path).astype(np.float32) / 255.0
    colorL_obj = cv2.imread(colorL_obj_path).astype(np.float32) / 255.0
    colorR_bg = cv2.imread(colorR_bg_path).astype(np.float32) / 255.0
    colorR_obj = cv2.imread(colorR_obj_path).astype(np.float32) / 255.0

    # Convert BGR to RGB for color extraction
    colorL_obj_rgb = cv2.cvtColor((colorL_obj * 255).astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    colorR_obj_rgb = cv2.cvtColor((colorR_obj * 255).astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Use compute_foreground_mask for robust mask (with largest component, morph ops)
    fg_maskL = compute_foreground_mask(colorL_bg, colorL_obj, threshold=mask_threshold)
    fg_maskR = compute_foreground_mask(colorR_bg, colorR_obj, threshold=mask_threshold)

    # Optional: Dilate the mask to fill small holes (tune kernel size/iterations as needed)
    dilation_kernel = np.ones(dilation_kernel_size, np.uint8)
    fg_maskL = cv2.dilate(fg_maskL.astype(np.uint8), dilation_kernel, iterations=dilation_iterations).astype(bool)
    fg_maskR = cv2.dilate(fg_maskR.astype(np.uint8), dilation_kernel, iterations=dilation_iterations).astype(bool)

    # Visualize foreground masks
    if visualize_mask:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.title('Foreground Mask L')
        plt.imshow(fg_maskL, cmap='gray')
        plt.subplot(1,2,2)
        plt.title('Foreground Mask R')
        plt.imshow(fg_maskR, cmap='gray')
        plt.show()

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

    # Visualize combined masks
    if visualize_mask:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.title('Combined Mask L')
        plt.imshow(maskL, cmap='gray')
        plt.subplot(1,2,2)
        plt.title('Combined Mask R')
        plt.imshow(maskR, cmap='gray')
        plt.show()

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

    # 6. Extract color for each triangulated point (use RGB)
    colorsL = colorL_obj_rgb[validL_y[matchL], validL_x[matchL]].T  # shape (3, N)
    colorsR = colorR_obj_rgb[validR_y[matchR], validR_x[matchR]].T
    colors = (colorsL + colorsR) / 2.0

    # 7. Triangulate the points
    pts3 = triangulate(pts2L, camL, pts2R, camR)

    # 8. 3D outlier removal (bounding box or stddev filter)
    if pts3.shape[1] > 0:
        mean = np.mean(pts3, axis=1, keepdims=True)
        std = np.std(pts3, axis=1, keepdims=True)
        keep = np.all(np.abs(pts3 - mean) < outlier_stddev * std, axis=0)
        pts3 = pts3[:, keep]
        colors = colors[:, keep]
        pts2L = pts2L[:, keep]
        pts2R = pts2R[:, keep]
        print(f"[DEBUG] After 3D outlier removal: {pts3.shape[1]} points remain")

    return pts2L, pts2R, pts3, colors

def generate_and_save_reconstruction(scan_path, output_pickle, threshold=0.01, mask_threshold=20, dilation_kernel_size=(9,9), dilation_iterations=2, outlier_stddev=2.5, visualize_mask=False, colorL_bg_path=None, colorR_bg_path=None):
    """
    Run the full reconstruction pipeline for a scan and save results to a pickle file.

    Args:
        scan_path (str): Path to the scan directory (e.g., koi/grab_0)
        output_pickle (str): Path to output pickle file
        threshold (float): Decoding threshold for pattern decoding
        mask_threshold (int): Foreground mask threshold
        dilation_kernel_size (tuple): Dilation kernel size for mask cleanup
        dilation_iterations (int): Number of dilation iterations
        outlier_stddev (float): Stddev for 3D outlier removal
        visualize_mask (bool): Whether to visualize masks
        colorL_bg_path (str): Optional path to left background image
        colorR_bg_path (str): Optional path to right background image
    """
    import os
    import pickle
    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Always use grab_0 backgrounds unless overridden
    grab0_path = os.path.join(project_root, "koi", "grab_0")
    default_colorL_bg_path = os.path.join(grab0_path, "color_C0_00_u.png")
    default_colorR_bg_path = os.path.join(grab0_path, "color_C1_00_u.png")
    colorL_bg_path = colorL_bg_path or default_colorL_bg_path
    colorR_bg_path = colorR_bg_path or default_colorR_bg_path
    colorL_obj_path = os.path.join(scan_path, "color_C0_01_u.png")
    colorR_obj_path = os.path.join(scan_path, "color_C1_01_u.png")
    imprefixL = os.path.join(scan_path, "frame_C0_")
    imprefixR = os.path.join(scan_path, "frame_C1_")
    stereo_calib_path = os.path.join(project_root, "calib", "stereo_calibration.pickle")
    with open(stereo_calib_path, "rb") as f:
        stereo_calib = pickle.load(f)
    KL = stereo_calib["KL"]
    KR = stereo_calib["KR"]
    R = stereo_calib["R"]
    T = stereo_calib["T"]
    from camutils import Camera
    camL = Camera(
        f=KL[0,0],
        c=np.array([[KL[0,2]], [KL[1,2]]]),
        R=np.eye(3),
        t=np.zeros((3,1))
    )
    camR = Camera(
        f=KR[0,0],
        c=np.array([[KR[0,2]], [KR[1,2]]]),
        R=R,
        t=T.reshape(3,1)
    )
    # Run reconstruction
    pts2L, pts2R, pts3, colors = reconstruct(
        imprefixL, imprefixR, threshold, camL, camR,
        colorL_bg_path, colorL_obj_path, colorR_bg_path, colorR_obj_path,
        mask_threshold=mask_threshold, visualize_mask=visualize_mask
    )
    # Save results
    results = {
        'pts3': pts3,
        'colors': colors,
        'pts2L': pts2L,
        'pts2R': pts2R,
    }
    with open(output_pickle, 'wb') as f:
        pickle.dump(results, f)
    print(f"[INFO] Saved reconstruction results to {output_pickle}")
