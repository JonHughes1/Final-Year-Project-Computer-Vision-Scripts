import numpy as np
import cv2
import glob

# === CONFIGURATION ===
CHECKERBOARD = (6, 9)  # Number of internal corners (NOT number of squares!)
square_size = 23.07    # In mm (your measurement)

# Termination criteria for cornerSubPix refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare the known object points (3D world points)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size  # Scale to real-world size

# Arrays to store object points and image points from all images
objpoints = []  # 3D points
imgpoints = []  # 2D points

# Get list of calibration images
image_folder = 'checkerboard-old/*.jpg'
images = glob.glob(image_folder)  # Adjust pattern to match your images

img_shape = None  # capture image size after first successful detection

# === PROCESS IMAGES ===
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Warning: Couldn't load image {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        # Save the image size on first success
        if img_shape is None:
            img_shape = gray.shape[::-1]  # (width, height)

        # Refine corner locations
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners2)

        # Draw and show the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey()
    else:
        print(f"Checkerboard couldn't be found in {fname}")

cv2.destroyAllWindows()

if img_shape is None:
    raise ValueError("No valid images found for calibration!")

# === CAMERA CALIBRATION ===
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_shape, None, None
)

# === OUTPUT RESULTS ===
print("\n--- Calibration Results ---")
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist.ravel())

# === COMPUTE MEAN REPROJECTION ERROR ===
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print(f"\nTotal mean reprojection error: {mean_error / len(objpoints):.4f} pixels")
