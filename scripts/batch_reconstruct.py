"""
batch_reconstruct.py

This script runs the reconstruction pipeline for all scan directories in batch mode, saving the results as pickle files.
"""
import os
from reconstruct import generate_and_save_reconstruction

# Directory containing all grabs
koi_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'koi')
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
os.makedirs(output_dir, exist_ok=True)

# List all grab directories (ignore hidden and non-directories)
grabs = [d for d in os.listdir(koi_dir) if d.startswith('grab_') and os.path.isdir(os.path.join(koi_dir, d))]
grabs.sort()

for grab in grabs:
    scan_path = os.path.join(koi_dir, grab)
    output_pickle = os.path.join(output_dir, f'{grab}_reconstruction.pickle')
    print(f'Processing {grab}...')
    generate_and_save_reconstruction(scan_path, output_pickle)
    print(f'Done: {output_pickle}\n')

print('All grabs processed.') 