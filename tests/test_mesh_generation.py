import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
from mesh_generator import (
    process_reconstruction_pickle,
    batch_process_reconstructions,
    align_meshes
)

def test_single_reconstruction():
    """Test mesh generation for a single reconstruction."""
    print("\n=== Testing Single Reconstruction ===")
    
    # Get the first reconstruction pickle file
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
    pickle_files = [f for f in os.listdir(output_dir) if f.endswith('_reconstruction.pickle')]
    if not pickle_files:
        print("No reconstruction pickle files found!")
        return
    
    test_pickle = os.path.join(output_dir, pickle_files[0])
    test_output_dir = os.path.join(output_dir, 'test_meshes', 'single')
    
    print(f"Processing {pickle_files[0]}...")
    
    # Try different mesh generation parameters
    mesh = process_reconstruction_pickle(
        test_pickle,
        output_dir=test_output_dir,
        voxel_size=0.01,  # Adjust based on your point cloud density
        depth=8,          # Adjust based on desired detail level
        use_trimesh=True, # Use trimesh for visualization
        visualize=True,   # Show the mesh
        export_formats=['obj', 'stl']  # Export to multiple formats
    )
    
    print(f"Mesh generated and saved to {test_output_dir}")
    print("Available files:")
    for f in os.listdir(test_output_dir):
        print(f"  - {f}")

def test_batch_processing():
    """Test batch processing of all reconstructions."""
    print("\n=== Testing Batch Processing ===")
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
    test_output_dir = os.path.join(output_dir, 'test_meshes', 'batch')
    
    print("Processing all reconstructions...")
    
    batch_process_reconstructions(
        output_dir,
        output_dir=test_output_dir,
        voxel_size=0.01,
        depth=8,
        use_trimesh=True,
        visualize=True,
        export_formats=['obj', 'stl']
    )
    
    print(f"\nAll meshes generated and saved to {test_output_dir}")
    print("Available files:")
    for root, dirs, files in os.walk(test_output_dir):
        for f in files:
            print(f"  - {os.path.join(os.path.relpath(root, test_output_dir), f)}")

def test_mesh_alignment():
    """Test mesh alignment of multiple reconstructions."""
    print("\n=== Testing Mesh Alignment ===")
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
    mesh_dir = os.path.join(output_dir, 'test_meshes', 'batch')
    
    # Get all PLY files from the batch processing
    mesh_files = []
    for root, _, files in os.walk(mesh_dir):
        for f in files:
            if f.endswith('.ply'):
                mesh_files.append(os.path.join(root, f))
    
    if len(mesh_files) < 2:
        print("Need at least 2 meshes for alignment testing!")
        return
    
    print(f"Found {len(mesh_files)} meshes for alignment")
    print("Aligning meshes...")
    
    aligned_dir = os.path.join(output_dir, 'test_meshes', 'aligned')
    aligned_meshes = align_meshes(
        mesh_files,
        reference_idx=0,  # Use first mesh as reference
        output_dir=aligned_dir
    )
    
    print(f"\nAligned meshes saved to {aligned_dir}")
    print("Available files:")
    for f in os.listdir(aligned_dir):
        print(f"  - {f}")

def test_meshlab_cleaning():
    """Test mesh cleaning with Meshlab (if available)."""
    print("\n=== Testing Meshlab Cleaning ===")
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
    pickle_files = [f for f in os.listdir(output_dir) if f.endswith('_reconstruction.pickle')]
    if not pickle_files:
        print("No reconstruction pickle files found!")
        return
    
    test_pickle = os.path.join(output_dir, pickle_files[0])
    test_output_dir = os.path.join(output_dir, 'test_meshes', 'cleaned')
    
    print(f"Processing {pickle_files[0]} with Meshlab cleaning...")
    
    try:
        mesh = process_reconstruction_pickle(
            test_pickle,
            output_dir=test_output_dir,
            voxel_size=0.01,
            depth=8,
            use_trimesh=True,
            clean_with_meshlab=True,  # Try to use Meshlab
            visualize=True,
            export_formats=['obj', 'stl']
        )
        print(f"Mesh cleaned and saved to {test_output_dir}")
        print("Available files:")
        for f in os.listdir(test_output_dir):
            print(f"  - {f}")
    except Exception as e:
        print(f"Meshlab cleaning failed: {e}")
        print("Please ensure Meshlab is installed and meshlabserver is in your PATH")

def main():
    parser = argparse.ArgumentParser(description='Test mesh generation and processing.')
    parser.add_argument('--batch', action='store_true', help='Process all reconstruction pickles in the directory')
    parser.add_argument('pickle', nargs='?', default=None, help='Path to a specific reconstruction pickle file')
    parser.add_argument('--pickle_dir', default="outputs/test_meshes/single", help='Directory containing reconstruction pickles')
    parser.add_argument('--output_dir', default="outputs/test_meshes/single", help='Directory to save mesh outputs')
    args = parser.parse_args()

    if args.batch:
        print(f"Batch processing all pickles in {args.pickle_dir}...")
        batch_process_reconstructions(
            pickle_dir=args.pickle_dir,
            output_dir=args.output_dir
        )
        print("Batch processing complete.")
    elif args.pickle:
        print(f"Processing single pickle: {args.pickle}")
        process_reconstruction_pickle(
            pickle_path=args.pickle,
            output_dir=args.output_dir
        )
        print("Single mesh processing complete.")
    else:
        # Default: process the first pickle found
        pickles = [f for f in os.listdir(args.pickle_dir) if f.endswith('.pickle')]
        if not pickles:
            print(f"No pickle files found in {args.pickle_dir}")
            return
        first_pickle = os.path.join(args.pickle_dir, pickles[0])
        print(f"Processing first pickle found: {first_pickle}")
        process_reconstruction_pickle(
            pickle_path=first_pickle,
            output_dir=args.output_dir
        )
        print("Default single mesh processing complete.")

if __name__ == '__main__':
    main() 