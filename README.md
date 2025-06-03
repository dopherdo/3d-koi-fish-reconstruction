# 3D Object Reconstruction

This project implements a 3D reconstruction pipeline for capturing and reconstructing 3D objects using structured light and stereo vision. The system uses two calibrated cameras and a projector to capture the object's geometry and color information.

## Features

- Structured light pattern decoding
- Stereo camera calibration
- Foreground/background segmentation
- 3D point cloud reconstruction with color
- Visualization tools

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
│   └── koi/                      # Sample object scans
└── outputs/                      # Output reconstructions
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

[Your Name] - CS 117 Final Project 