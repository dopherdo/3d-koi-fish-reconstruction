import os
import pickle
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# --- User: Set these paths for the 5 scans you want to align ---
scans = [
    "grab_0_reconstruction",
    "grab_1_reconstruction",
    "grab_2_reconstruction",
    "grab_3_reconstruction",
    "grab_4_reconstruction"
]
base_dir = "outputs/test_meshes/single"
img_paths = [f"koi/grab_{i}/color_C0_01_u.png" for i in range(5)]
pickle_paths = [f"outputs/grab_{i}_reconstruction.pickle" for i in range(5)]
mesh_paths = [os.path.join(base_dir, scans[i], f"{scans[i]}_mesh_fixed_smoothed.ply") for i in range(5)]

n_points = 4  # Number of correspondences to click

# --- 1. Click corresponding points for all images in a row ---
def get_points_from_images(img_paths, n_points=4):
    n_scans = len(img_paths)
    pts2d = [np.zeros((2, n_points)) for _ in range(n_scans)]
    for pt_idx in range(n_points):
        fig, axs = plt.subplots(1, n_scans, figsize=(4*n_scans, 4))
        for i, img_path in enumerate(img_paths):
            img = plt.imread(img_path)
            axs[i].imshow(img)
            axs[i].set_title(f"Scan {i}: Click point {pt_idx+1}")
            axs[i].axis('off')
        plt.suptitle(f"Click correspondence {pt_idx+1} in each image (left to right)")
        clicks = plt.ginput(n_scans, timeout=0)
        plt.close()
        for i, (x, y) in enumerate(clicks):
            pts2d[i][:, pt_idx] = [x, y]
    return pts2d  # List of (2, n_points) arrays

print(f"Click {n_points} corresponding points in each of the 5 images (left to right)...")
pts2d_list = get_points_from_images(img_paths, n_points=n_points)

# --- 2. Map 2D clicks to nearest pts2L, get 3D points for each scan ---
def get_3d_points(pickle_path, pts2d_clicks):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    pts2L = data['pts2L']  # shape (2, N)
    pts3 = data['pts3']    # shape (3, N)
    tree = cKDTree(pts2L.T)
    dists, idxs = tree.query(pts2d_clicks.T)
    pts3_sel = pts3[:, idxs]  # shape (3, n_clicks)
    return pts3_sel

pts3_list = [get_3d_points(pickle_paths[i], pts2d_list[i]) for i in range(5)]

# --- 3. Compute SVD alignment (Kabsch) to align all to the first scan ---
def svd_align(A, B):
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA @ BB.T
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    return R, t

# Compute transforms to align each scan to scan 0
transforms = [np.eye(4)]  # First scan is reference
for i in range(1, 5):
    R, t = svd_align(pts3_list[i], pts3_list[0])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:] = t
    transforms.append(T)
    print(f"Alignment for scan {i} to scan 0:")
    print("Rotation:\n", R)
    print("Translation:\n", t.flatten())

# --- 4. Apply transformations and merge meshes ---
meshes = []
for i in range(5):
    mesh = o3d.io.read_triangle_mesh(mesh_paths[i])
    mesh.transform(transforms[i])
    meshes.append(mesh)

merged = meshes[0]
for mesh in meshes[1:]:
    merged += mesh

out_path = os.path.join(base_dir, "all_scans_aligned_merged.ply")
o3d.io.write_triangle_mesh(out_path, merged)
print(f"Saved merged mesh: {out_path}")

# Visualize merged mesh
o3d.visualization.draw_geometries([merged]) 