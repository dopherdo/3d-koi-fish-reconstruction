# 3D Koi Fish Reconstruction

This project implements a 3D reconstruction pipeline for capturing and reconstructing a koi fish model using structured light and stereo vision. The system uses two calibrated cameras and a projector to capture the object's geometry and color information, with a focus on preserving the intricate details and colors of the koi fish.

## Features

- Structured light pattern decoding for precise 3D capture
- Stereo camera calibration for accurate depth estimation
- Foreground/background segmentation using color images
- 3D point cloud reconstruction with color preservation
- Visualization tools for inspecting the reconstruction

## Project Structure

```
.
├── README.md
├── requirements.txt
├── scripts/
│   ├── reconstruct_modified.py    # Main reconstruction pipeline
│   ├── test_reconstruction.py     # Testing and visualization
│   └── meshutils.py              # Mesh utilities
├── calib/                        # Camera calibration data
│   ├── calib_C0.pickle
│   └── calib_C1.pickle
├── data/                         # Sample data (not tracked in git)
│   └── koi/                      # Koi fish scan data
└── outputs/                      # Reconstruction outputs
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Calibrate cameras (if needed):
```bash
python scripts/test_calibration_and_save.py
```

2. Run reconstruction:
```bash
python scripts/test_reconstruction.py
```

## Dependencies

- Python 3.8+
- OpenCV
- NumPy
- Matplotlib
- See requirements.txt for full list

## License

MIT License - see LICENSE file for details

## Author

Chris Yeh - CS 117 Final Project 