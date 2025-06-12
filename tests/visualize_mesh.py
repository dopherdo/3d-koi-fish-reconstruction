"""
visualize_mesh.py

This script is a utility for rendering point clouds and meshes using Open3D. Helps visualize intermediate results such as raw point clouds, aligned scans, and final meshes.
"""
import sys
import os
import open3d as o3d

# Default mesh path
DEFAULT_MESH = os.path.join('outputs', 'meshes', 'grab_1', 'mesh.ply')

def main():
    if len(sys.argv) > 1:
        mesh_path = sys.argv[1]
    else:
        mesh_path = DEFAULT_MESH
        if not os.path.exists(mesh_path):
            mesh_path = input('Enter path to mesh file (.ply, .obj, .stl): ').strip()

    if not os.path.exists(mesh_path):
        print(f"Mesh file not found: {mesh_path}")
        return

    print(f"Loading mesh: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_triangles():
        print("Loaded mesh has no triangles!")
        return
    print(mesh)
    o3d.visualization.draw_geometries([mesh], window_name=f"Mesh Viewer: {os.path.basename(mesh_path)}")

if __name__ == '__main__':
    main() 