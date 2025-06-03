import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from reconstruct_modified import reconstruct
import cv2
import os

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
    scan_path = "../koi/grab_0"
    calib_path0 = "../calib/calib_C0.pickle"
    calib_path1 = "../calib/calib_C1.pickle"
    
    # Load and decode pattern images
    print("Loading pattern images...")
    pattern_images0 = [cv2.imread(os.path.join(scan_path, f"frame_C0_{i:02d}_u.png"), cv2.IMREAD_GRAYSCALE) 
                      for i in range(40)]
    pattern_images1 = [cv2.imread(os.path.join(scan_path, f"frame_C1_{i:02d}_u.png"), cv2.IMREAD_GRAYSCALE) 
                      for i in range(40)]
    
    # Decode patterns (placeholder)
    print("Decoding patterns...")
    uv0, mask0 = decode_pattern(pattern_images0)
    uv1, mask1 = decode_pattern(pattern_images1)
    
    # Run reconstruction
    print("Running reconstruction...")
    pts3, colors, mask = reconstruct(
        scan_path=scan_path,
        valid_mask0=mask0,
        valid_mask1=mask1,
        uv0=uv0,
        uv1=uv1,
        calib_path0=calib_path0,
        calib_path1=calib_path1,
        visualize_mask=True
    )
    
    # Print some statistics
    print("\nReconstruction Statistics:")
    print(f"Number of 3D points: {pts3.shape[1]}")
    print(f"Point cloud bounds:")
    print(f"  X: [{pts3[0].min():.2f}, {pts3[0].max():.2f}]")
    print(f"  Y: [{pts3[1].min():.2f}, {pts3[1].max():.2f}]")
    print(f"  Z: [{pts3[2].min():.2f}, {pts3[2].max():.2f}]")
    
    # Visualize the point cloud
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points with their colors
    scatter = ax.scatter(pts3[0], pts3[1], pts3[2], 
                        c=colors.T,  # colors should be Nx3
                        s=1,         # point size
                        alpha=0.5)   # transparency
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Reconstruction with Color')
    
    # Equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main() 