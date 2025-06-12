import os
import open3d as o3d

# Directory containing subfolders for each grab's mesh
base_dir = "outputs/test_meshes/single"

# Number of smoothing iterations
num_iterations = 5

# Loop through all subdirectories
for grab_dir in os.listdir(base_dir):
    grab_path = os.path.join(base_dir, grab_dir)
    if not os.path.isdir(grab_path):
        continue
    mesh_path = os.path.join(grab_path, f"{grab_dir}_mesh_fixed.ply")
    if not os.path.exists(mesh_path):
        print(f"Fixed mesh not found: {mesh_path}")
        continue

    print(f"Smoothing fixed mesh: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    smoothed_mesh = mesh.filter_smooth_simple(number_of_iterations=num_iterations)
    smoothed_path = os.path.join(grab_path, f"{grab_dir}_mesh_fixed_smoothed.ply")
    o3d.io.write_triangle_mesh(smoothed_path, smoothed_mesh)
    print(f"Saved smoothed mesh: {smoothed_path}") 