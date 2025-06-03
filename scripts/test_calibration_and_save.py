import pickle
import numpy as np
import cv2
import glob
import os

def calibrate_camera(images, pattern_size=(8, 6), square_size=2.8):
    """Calibrate a camera given checkerboard images."""
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = square_size * np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"⚠️ Could not read image: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)
            cv2.imshow('Checkerboard', img)
            cv2.waitKey(300)
        else:
            print(f"❌ Chessboard not found in: {fname}")

    cv2.destroyAllWindows()

    if not objpoints:
        raise ValueError("No valid checkerboard detections for calibration.")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("✅ Calibration Successful")
    print("Camera Matrix (K):\n", K)
    print("Distortion Coefficients:\n", dist.ravel())

    return {
        "K": K,
        "dist": dist,
        "fx": K[0][0],
        "fy": K[1][1],
        "cx": K[0][2],
        "cy": K[1][2],
        "rvecs": rvecs,
        "tvecs": tvecs
    }


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    calib_dir = os.path.join(script_dir, "../calib")

    image_glob1 = os.path.join(calib_dir, "frame_C0_*_u.png")
    image_glob2 = os.path.join(calib_dir, "frame_C1_*_u.png")

    image_paths1 = sorted(glob.glob(image_glob1))
    print(f"📸 Found {len(image_paths1)} images for C0 using pattern: {image_glob1}")
    if image_paths1:
        save_path1 = os.path.join(calib_dir, "calib_C0.pickle")
        calib_result1 = calibrate_camera(image_paths1)
        with open(save_path1, "wb") as f:
            pickle.dump(calib_result1, f)
        print("✔️ Calibration done for C0")
    else:
        print("⚠️ No images found for C0, skipping.")

    image_paths2 = sorted(glob.glob(image_glob2))
    print(f"📸 Found {len(image_paths2)} images for C1 using pattern: {image_glob2}")
    if image_paths2:
        save_path2 = os.path.join(calib_dir, "calib_C1.pickle")
        calib_result2 = calibrate_camera(image_paths2)
        with open(save_path2, "wb") as f:
            pickle.dump(calib_result2, f)
        print("✔️ Calibration done for C1")
    else:
        print("⚠️ No images found for C1, skipping.")