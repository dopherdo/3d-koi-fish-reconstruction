import cv2
import numpy as np
import os
import pickle

def compute_foreground_mask(bg_img, obj_img, threshold=30, visualize=False):
    diff = np.abs(obj_img - bg_img)
    diff_gray = cv2.cvtColor((diff * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
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

def triangulate_points(uv0, uv1, mask, calib0, calib1):
    fx0, fy0, cx0, cy0 = calib0['fx'], calib0['fy'], calib0['cx'], calib0['cy']
    fx1, fy1, cx1, cy1 = calib1['fx'], calib1['fy'], calib1['cx'], calib1['cy']
    K0 = np.array([[fx0, 0, cx0], [0, fy0, cy0], [0, 0, 1]])
    K1 = np.array([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]])
    P0 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = np.hstack((np.eye(3), np.array([[1], [0], [0]])))
    P0 = K0 @ P0
    P1 = K1 @ P1
    uv0_h = cv2.undistortPoints(uv0.T.reshape(-1, 1, 2), K0, calib0['dist'])
    uv1_h = cv2.undistortPoints(uv1.T.reshape(-1, 1, 2), K1, calib1['dist'])
    pts4d = cv2.triangulatePoints(P0, P1, uv0_h, uv1_h)
    pts3d = (pts4d[:3] / pts4d[3]).reshape(3, -1)
    valid = validate_points(pts3d, P0, P1, uv0, uv1, mask)
    return pts3d[:, valid], valid

def reconstruct(scan_path, valid_mask0, valid_mask1, uv0, uv1, calib_path0, calib_path1, visualize_mask=False):
    with open(calib_path0, 'rb') as f0, open(calib_path1, 'rb') as f1:
        calib0, calib1 = pickle.load(f0), pickle.load(f1)

    bg0, obj0 = load_color_images(scan_path, "C0")
    bg1, obj1 = load_color_images(scan_path, "C1")

    fg_mask0 = compute_foreground_mask(bg0, obj0, visualize=visualize_mask)
    fg_mask1 = compute_foreground_mask(bg1, obj1, visualize=visualize_mask)

    mask0 = apply_combined_mask(valid_mask0, fg_mask0)
    mask1 = apply_combined_mask(valid_mask1, fg_mask1)
    final_mask = np.logical_and(mask0, mask1)

    pts3, valid = triangulate_points(uv0, uv1, final_mask, calib0, calib1)
    colors = extract_colors(obj0, obj1, final_mask[valid])

    return pts3, colors, final_mask[valid]
