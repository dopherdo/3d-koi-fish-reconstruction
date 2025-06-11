import cv2
import numpy as np
import glob
import pickle
import os

'''
Load chessboard images for both cameras (from calib/frame_C0_XX_u.png and calib/frame_C1_XX_u.png)
Find chessboard corners in both sets
Use OpenCV’s cv2.stereoCalibrate to compute the stereo extrinsics (R, T) and updated intrinsics/distortion
Save all calibration results (including R, T) to calib/stereo_calibration.pickle
Print a summary of the calibration results
'''

# Parameters
pattern_size = (8, 6)
square_size = 2.8
calib_dir = os.path.join(os.path.dirname(__file__), '../calib')
left_pattern = os.path.join(calib_dir, 'frame_C0_*_u.png')
right_pattern = os.path.join(calib_dir, 'frame_C1_*_u.png')
output_file = os.path.join(calib_dir, 'stereo_calibration.pickle')

# Prepare object points
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = square_size * np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# Find all left and right images
left_images = sorted(glob.glob(left_pattern))
right_images = sorted(glob.glob(right_pattern))

if len(left_images) != len(right_images):
    print(f"Number of left and right images do not match: {len(left_images)} vs {len(right_images)}")
    exit(1)

objpoints = []
imgpointsL = []
imgpointsR = []

for fnameL, fnameR in zip(left_images, right_images):
    imgL = cv2.imread(fnameL)
    imgR = cv2.imread(fnameR)
    if imgL is None or imgR is None:
        print(f"Could not read image pair: {fnameL}, {fnameR}")
        continue
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    retL, cornersL = cv2.findChessboardCorners(grayL, pattern_size, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, pattern_size, None)
    if retL and retR:
        objpoints.append(objp)
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)
        # Optionally show corners
        # cv2.drawChessboardCorners(imgL, pattern_size, cornersL, retL)
        # cv2.drawChessboardCorners(imgR, pattern_size, cornersR, retR)
        # cv2.imshow('L', imgL)
        # cv2.imshow('R', imgR)
        # cv2.waitKey(100)
    else:
        print(f"Chessboard not found in pair: {fnameL}, {fnameR}")

# cv2.destroyAllWindows()

if len(objpoints) < 3:
    print("Not enough valid pairs for stereo calibration.")
    exit(1)

# Calibrate each camera individually
retL, KL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
retR, KR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)

# Stereo calibration
flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
ret, KL, distL, KR, distR, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    KL, distL, KR, distR, grayL.shape[::-1],
    criteria=criteria, flags=flags)

print("Stereo calibration RMS error:", ret)
print("Left camera matrix:\n", KL)
print("Right camera matrix:\n", KR)
print("Left distortion:\n", distL.ravel())
print("Right distortion:\n", distR.ravel())
print("Rotation (R):\n", R)
print("Translation (T):\n", T.ravel())

# Save calibration results
calib = {
    'KL': KL, 'distL': distL,
    'KR': KR, 'distR': distR,
    'R': R, 'T': T,
    'E': E, 'F': F
}
with open(output_file, 'wb') as f:
    pickle.dump(calib, f)
print(f"Stereo calibration saved to {output_file}") 