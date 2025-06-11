import numpy as np
import open3d as o3d
import trimesh
import pickle
import os
import subprocess
from meshutils import writeply
from pathlib import Path

def convert_mesh_format(input_path, output_path, format='ply'):
    """
    Convert mesh between different formats using trimesh.
    
    Parameters
    ----------
    input_path : str
        Path to input mesh file
    output_path : str
        Path to output mesh file
    format : str, optional
        Output format (ply, obj, stl, etc.)
    """
    mesh = trimesh.load(input_path)
    mesh.export(output_path, file_type=format)

def clean_mesh_with_meshlab(input_path, output_path, script_path=None):
    """
    Clean mesh using Meshlab's command-line interface.
    If script_path is provided, use that script for cleaning.
    Otherwise, apply basic cleaning operations.
    
    Parameters
    ----------
    input_path : str
        Path to input mesh file
    output_path : str
        Path to output mesh file
    script_path : str, optional
        Path to Meshlab script file (.mlx)
    """
    if script_path and os.path.exists(script_path):
        cmd = ['meshlabserver', '-i', input_path, '-o', output_path, '-s', script_path]
    else:
        # Basic cleaning script
        temp_script = 'temp_clean.mlx'
        with open(temp_script, 'w') as f:
            f.write("""
<!DOCTYPE FilterScript>
<FilterScript>
 <filter name="Remove Duplicate Vertices"/>
 <filter name="Remove Duplicate Faces"/>
 <filter name="Remove Unreferenced Vertices"/>
 <filter name="Remove Zero Area Faces"/>
 <filter name="Remove Non Manifold Edges"/>
 <filter name="Close Holes"/>
</FilterScript>
            """)
        cmd = ['meshlabserver', '-i', input_path, '-o', output_path, '-s', temp_script]
        try:
            subprocess.run(cmd, check=True)
            os.remove(temp_script)
        except subprocess.CalledProcessError as e:
            print(f"Error running Meshlab: {e}")
            print("Please ensure Meshlab is installed and meshlabserver is in your PATH")
            raise

def generate_mesh_from_point_cloud(
    pts3,
    colors=None,
    output_pickle=None,
    output_mesh=None,
    output_pointcloud=None,
    voxel_size=0.01,
    depth=8,
    width=0,
    scale=1.1,
    linear_fit=False,
    visualize=False,
    use_trimesh=False,
    clean_with_meshlab=False,
    meshlab_script=None
):
    """
    Generate a mesh from a point cloud using Poisson surface reconstruction.
    Optionally saves intermediate data and visualizes the result.
    Args:
        pts3: (N, 3) numpy array of 3D points
        colors: (N, 3) numpy array of colors (optional)
        output_pickle: path to save intermediate data (optional)
        output_mesh: path to save the mesh (optional)
        output_pointcloud: path to save the point cloud as .ply (optional)
        voxel_size: size of voxels for Poisson reconstruction
        depth: depth of the octree used for Poisson reconstruction
        width: width parameter for Poisson reconstruction
        scale: scale parameter for Poisson reconstruction
        linear_fit: whether to use linear interpolation for Poisson reconstruction
        visualize: whether to visualize the point cloud and mesh
        use_trimesh: whether to use trimesh for visualization
        clean_with_meshlab: whether to clean the mesh using Meshlab
        meshlab_script: path to Meshlab script for cleaning
    """
    # Convert point cloud to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3.T)
    pcd.colors = o3d.utility.Vector3dVector(colors.T)
    
    # Estimate normals if not present
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(100)
    
    # Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit
    )
    
    # Optional: Remove low-density vertices
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # Optional: Clean up the mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    # Save intermediate data if requested
    if output_pickle:
        # Only save serializable data (numpy arrays)
        pickle.dump({
            'pts3': pts3,
            'colors': colors
        }, open(output_pickle, 'wb'))
    if output_mesh:
        o3d.io.write_triangle_mesh(str(output_mesh), mesh, write_ascii=True)
    if output_pointcloud:
        # Save the point cloud as a .ply file
        o3d.io.write_point_cloud(str(output_pointcloud), pcd, write_ascii=True)
    
    # Convert to trimesh if requested
    if use_trimesh:
        # Convert Open3D mesh to trimesh
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        if hasattr(mesh, 'visual'):
            mesh.visual.vertex_colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
    
    # Optional: Visualize
    if visualize:
        if use_trimesh:
            mesh.show()
        else:
            o3d.visualization.draw_geometries([mesh])
    
    return mesh

def process_reconstruction_pickle(
    pickle_path,
    output_dir=None,
    voxel_size=0.01,
    depth=8,
    width=0,
    scale=1.1,
    linear_fit=False,
    visualize=False,
    use_trimesh=False,
    clean_with_meshlab=False,
    meshlab_script=None,
    export_formats=None,
    output_pointcloud=None
):
    """
    Process a reconstruction pickle file and generate a mesh.
    Args:
        pickle_path: path to the reconstruction pickle
        output_dir: directory to save outputs
        voxel_size: size of voxels for Poisson reconstruction
        depth: depth of the octree used for Poisson reconstruction
        width: width parameter for Poisson reconstruction
        scale: scale parameter for Poisson reconstruction
        linear_fit: whether to use linear interpolation for Poisson reconstruction
        visualize: whether to visualize the point cloud and mesh
        use_trimesh: whether to use trimesh for visualization
        clean_with_meshlab: whether to clean the mesh using Meshlab
        meshlab_script: path to Meshlab script for cleaning
        export_formats: list of additional formats to export the mesh to (e.g., ['obj', 'stl'])
        output_pointcloud: path to save the point cloud as .ply (optional)
    """
    # Load reconstruction data
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    pts3 = data['pts3']
    colors = data['colors']
    
    # Generate output paths if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(pickle_path))[0]
        output_pickle = os.path.join(output_dir, f'{base_name}_pointcloud.pickle')
        output_mesh = os.path.join(output_dir, f'{base_name}_mesh.ply')
    else:
        output_pickle = None
        output_mesh = None
    
    # Generate mesh
    mesh = generate_mesh_from_point_cloud(
        pts3=pts3,
        colors=colors,
        output_pickle=output_pickle,
        output_mesh=output_mesh,
        output_pointcloud=output_pointcloud,
        voxel_size=voxel_size,
        depth=depth,
        width=width,
        scale=scale,
        linear_fit=linear_fit,
        visualize=visualize,
        use_trimesh=use_trimesh,
        clean_with_meshlab=clean_with_meshlab,
        meshlab_script=meshlab_script
    )
    
    # Export to additional formats if requested
    if export_formats and output_mesh:
        for fmt in export_formats:
            output_path = str(Path(output_mesh).with_suffix(f'.{fmt}'))
            if isinstance(mesh, trimesh.Trimesh):
                mesh.export(output_path)
            else:
                convert_mesh_format(output_mesh, output_path, format=fmt)
    
    return mesh

def batch_process_reconstructions(pickle_dir, output_dir=None, **kwargs):
    """
    Process all reconstruction pickle files in a directory.
    
    Parameters
    ----------
    pickle_dir : str
        Directory containing reconstruction pickle files
    output_dir : str, optional
        Directory to save output files
    **kwargs : dict
        Additional parameters to pass to process_reconstruction_pickle
    """
    pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('_reconstruction.pickle')]
    pickle_files.sort()
    
    for pf in pickle_files:
        print(f'\nProcessing {pf}...')
        pickle_path = os.path.join(pickle_dir, pf)
        if output_dir:
            scan_output_dir = os.path.join(output_dir, os.path.splitext(pf)[0])
        else:
            scan_output_dir = None
        
        try:
            mesh = process_reconstruction_pickle(
                pickle_path,
                output_dir=scan_output_dir,
                **kwargs
            )
            print(f'Successfully generated mesh for {pf}')
        except Exception as e:
            print(f'Error processing {pf}: {str(e)}')
    
    print('\nAll reconstructions processed.')

def align_meshes(mesh_paths, reference_idx=0, output_dir=None):
    """
    Align multiple meshes using ICP (Iterative Closest Point) algorithm.
    
    Parameters
    ----------
    mesh_paths : list
        List of paths to mesh files to align
    reference_idx : int, optional
        Index of the reference mesh (others will be aligned to this one)
    output_dir : str, optional
        Directory to save aligned meshes
        
    Returns
    -------
    aligned_meshes : list
        List of aligned trimesh meshes
    """
    if not mesh_paths:
        return []
    
    # Load all meshes
    meshes = [trimesh.load(path) for path in mesh_paths]
    reference = meshes[reference_idx]
    aligned = [reference]
    
    # Align each mesh to the reference
    for i, mesh in enumerate(meshes):
        if i == reference_idx:
            continue
            
        # Compute alignment
        matrix, cost = trimesh.registration.mesh_other(mesh, reference)
        mesh.apply_transform(matrix)
        aligned.append(mesh)
        
        # Save aligned mesh if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'aligned_{os.path.basename(mesh_paths[i])}')
            mesh.export(output_path)
    
    return aligned 