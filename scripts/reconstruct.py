import cv2
import numpy as np
import os

def compute_foreground_mask(bg_img, obj_img, threshold=30, visualize=False):
    """
    Compute a binary foreground mask by subtracting the background from the object image.
    
    Parameters:
        bg_img (ndarray): Background color image.
        obj_img (ndarray): Object color image.
        threshold (int): Threshold for difference binarization.
        visualize (bool): Whether to show the mask visually.
    
    Returns:
        np.ndarray: Boolean mask indicating foreground pixels.
    """
    print("bg_img shape:", bg_img.shape)
    print("obj_img shape:", obj_img.shape)

    diff = np.abs(obj_img - bg_img)
    diff_gray = cv2.cvtColor((diff * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)

    if visualize:
        cv2.imshow("Foreground Mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mask.astype(bool)


def load_color_images(base_path, cam="C0"):
    """
    Load the background and object color images for the given camera.
    
    Parameters:
        base_path (str): Path to directory containing images.
        cam (str): Camera ID, e.g., "C0" or "C1".
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (background image, object image)
    
    Raises:
        FileNotFoundError: If one or both images could not be loaded.
    """
    bg_path = os.path.join(base_path, f"frame_{cam}_00_u.png")
    obj_path = os.path.join(base_path, f"frame_{cam}_01_u.png")

    bg = cv2.imread(bg_path)
    obj = cv2.imread(obj_path)

    if bg is None or obj is None:
        raise FileNotFoundError(f"Could not load images:\n  {bg_path}\n  {obj_path}")

    # Resize both to 1080p to match valid_mask shape
    target_size = (1920, 1080)
    bg = cv2.resize(bg, target_size, interpolation=cv2.INTER_AREA)
    obj = cv2.resize(obj, target_size, interpolation=cv2.INTER_AREA)

    bg = bg.astype(np.float32) / 255.0
    obj = obj.astype(np.float32) / 255.0
    return bg, obj


def apply_combined_mask(valid_mask, fg_mask):
    """
    Combine decoding and foreground masks.
    
    Returns:
        np.ndarray: Combined boolean mask.
    """
    return np.logical_and(valid_mask, fg_mask)


def extract_colors(obj_img, mask):
    """
    Extract RGB values from the object image where the mask is True.
    
    Parameters:
        obj_img (ndarray): Color image of the object.
        mask (ndarray): Boolean mask.
    
    Returns:
        np.ndarray: 3xN RGB values.
    """
    color_array = obj_img[mask]  # shape (N, 3)
    return color_array.T         # shape (3, N)


def triangulate_points(mask):
    """
    Placeholder for stereo triangulation. Currently returns dummy 3D points.
    
    Parameters:
        mask (ndarray): Boolean mask indicating which pixels to triangulate.
    
    Returns:
        np.ndarray: 3xN array of 3D points.
    """
    # TODO: Replace this with actual triangulation using decoded (u,v) and calibration
    pts3 = np.random.rand(3, np.sum(mask))  # Dummy points
    return pts3


def reconstruct(scan_path, valid_mask, cam="C0", visualize_mask=False):
    """
    Perform reconstruction by computing foreground, combining masks, and triangulating.
    
    Parameters:
        scan_path (str): Path to folder with scan images.
        valid_mask (np.ndarray): Mask of valid decoded pixels.
        cam (str): Camera ID (e.g. "C0", "C1").
        visualize_mask (bool): Whether to visualize the foreground mask.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (3D points, RGB colors, combined mask)
    """
    print("▶ Loading color images...")
    bg_img, obj_img = load_color_images(scan_path, cam=cam)
    
    print("▶ Shapes:")
    print("  bg_img:", bg_img.shape)
    print("  obj_img:", obj_img.shape)
    
    print("▶ Computing foreground mask...")
    fg_mask = compute_foreground_mask(bg_img, obj_img, visualize=visualize_mask)
    
    print("▶ fg_mask shape:", fg_mask.shape)
    # Step 3: Combine with decoding mask
    combined_mask = apply_combined_mask(valid_mask, fg_mask)

    # Step 4: Extract RGB values
    colors = extract_colors(obj_img, combined_mask)

    # Step 5: Triangulate 3D points (stub)
    pts3 = triangulate_points(combined_mask)

    return pts3, colors, combined_mask
