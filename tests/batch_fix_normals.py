import os
import open3d as o3d
import numpy as np

# Directory containing subfolders for each grab's mesh
base_dir = "outputs/test_meshes/single"

# Loop through all subdirectories
for grab_dir in os.listdir(base_dir):
    grab_path = os.path.join(base_dir, grab_dir)
    if not os.path.isdir(grab_path):
        continue
    mesh_path = os.path.join(grab_path, f"{grab_dir}_mesh.ply")
    if not os.path.exists(mesh_path):
        print(f"Mesh not found: {mesh_path}")
        continue

    print(f"Fixing normals for mesh: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh.orient_triangles()
    # Flip normals using numpy arrays
    mesh.triangle_normals = o3d.utility.Vector3dVector(-np.asarray(mesh.triangle_normals))
    mesh.vertex_normals = o3d.utility.Vector3dVector(-np.asarray(mesh.vertex_normals))
    fixed_path = os.path.join(grab_path, f"{grab_dir}_mesh_fixed.ply")
    o3d.io.write_triangle_mesh(fixed_path, mesh)
    print(f"Saved fixed mesh: {fixed_path}") 