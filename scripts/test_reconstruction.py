import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from reconstruct import reconstruct
import cv2
import os
from camutils import Camera

def decode_pattern(images):
    """
    Simple placeholder for pattern decoding.
    Returns dummy decoded points and mask for testing.
    """
    # Create a dummy mask (center region of image)
    mask = np.zeros((1080, 1920), dtype=bool)
    center_y, center_x = 1080//2, 1920//2
    mask[center_y-200:center_y+200, center_x-200:center_x+200] = True
    
    # Create dummy decoded points (grid in the masked region)
    y, x = np.where(mask)
    points = np.vstack((x, y))
    
    return points, mask

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    scan_path = os.path.join(project_root, "koi", "grab_0")
    calib_path0 = os.path.join(project_root, "calib", "calib_C0.pickle")
    calib_path1 = os.path.join(project_root, "calib", "calib_C1.pickle")

    # Color image paths
    colorL_bg_path = os.path.join(scan_path, "color_C0_00_u.png")
    colorL_obj_path = os.path.join(scan_path, "color_C0_01_u.png")
    colorR_bg_path = os.path.join(scan_path, "color_C1_00_u.png")
    colorR_obj_path = os.path.join(scan_path, "color_C1_01_u.png")

    # Pattern image prefixes
    imprefixL = os.path.join(scan_path, "frame_C0_")
    imprefixR = os.path.join(scan_path, "frame_C1_")
    threshold = 0.01  # raised threshold for decoding

    # Load calibration objects (replace with your Camera class if needed)
    with open(calib_path0, "rb") as f:
        calibL = pickle.load(f)
    with open(calib_path1, "rb") as f:
        calibR = pickle.load(f)

    # Load stereo calibration results
    stereo_calib_path = os.path.join(project_root, "calib", "stereo_calibration.pickle")
    with open(stereo_calib_path, "rb") as f:
        stereo_calib = pickle.load(f)

    KL = stereo_calib["KL"]
    distL = stereo_calib["distL"]
    KR = stereo_calib["KR"]
    distR = stereo_calib["distR"]
    R = stereo_calib["R"]
    T = stereo_calib["T"]

    # Camera objects
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

    # Call the new reconstruct function
    pts2L, pts2R, pts3, colors = reconstruct(
        imprefixL, imprefixR, threshold, camL, camR,
        colorL_bg_path, colorL_obj_path, colorR_bg_path, colorR_obj_path
    )

    print(f"[DEBUG] Number of matched correspondences: {pts2L.shape[1]}")

    # Print some statistics
    print("\nReconstruction Statistics:")
    print(f"Number of 3D points: {pts3.shape[1]}")
    if pts3.shape[1] == 0:
        print("No 3D points were reconstructed. Check your masks and decoding.")
        return
    print(f"Point cloud bounds:")
    print(f"  X: [{pts3[0].min():.2f}, {pts3[0].max():.2f}]")
    print(f"  Y: [{pts3[1].min():.2f}, {pts3[1].max():.2f}]")
    print(f"  Z: [{pts3[2].min():.2f}, {pts3[2].max():.2f}]")

    # Visualize the point cloud with color
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts3[0], pts3[1], pts3[2], c=colors.T, s=1, alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Reconstruction with Color')
    ax.set_box_aspect([1,1,1])
    plt.show()

if __name__ == "__main__":
    main() 