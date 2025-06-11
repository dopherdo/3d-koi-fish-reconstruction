import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
import pickle
import numpy as np
import matplotlib.pyplot as plt

output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
pickle_files = [f for f in os.listdir(output_dir) if f.endswith('_reconstruction.pickle')]
pickle_files.sort()

for pf in pickle_files:
    path = os.path.join(output_dir, pf)
    print(f'\nChecking {pf}...')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    pts3 = data['pts3']
    colors = data['colors']
    npts = pts3.shape[1]
    print(f'  Number of points: {npts}')
    if npts == 0:
        print('  [WARNING] No valid points in this scan!')
        continue
    print(f'  X range: {np.min(pts3[0]):.2f} to {np.max(pts3[0]):.2f}')
    print(f'  Y range: {np.min(pts3[1]):.2f} to {np.max(pts3[1]):.2f}')
    print(f'  Z range: {np.min(pts3[2]):.2f} to {np.max(pts3[2]):.2f}')
    print(f'  Color min/max: {np.min(colors):.2f} to {np.max(colors):.2f}')
    # Optional: visualize
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts3[0], pts3[1], pts3[2], c=colors.T, s=1, alpha=0.5)
    ax.set_title(pf)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

print('\nAll pickle files checked.') 