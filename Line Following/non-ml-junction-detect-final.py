from pypylon import pylon
import cv2
import numpy as np
import numpy as np
from skimage.morphology import skeletonize
import math
import socket
import time

# Create function for sending data to the pi
PI_IP = "192.168.10.2"   # Replace with the Pi's actual IP
PORT = 65432

def send_numbers(x, y):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((PI_IP, PORT))
        s.sendall(f"{x},{y}\n".encode())
        
# Initialize the camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Open the camera
camera.Open()

# Set exposure
camera.ExposureTime.SetValue(25000)

# Set camera to continuous grabbing mode
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

# Create an image converter
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_RGB8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# Grab frames
while camera.IsGrabbing():
    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grab_result.GrabSucceeded():
        # Convert the image to an OpenCV format
        image = converter.Convert(grab_result)
        img_array = image.GetArray()

        # Display the image
        cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Feed", 800, 600)
        cv2.imshow("Camera Feed", img_array)
        
        # Junction detection processing here:
        img = img_array
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        # Sobel filter for horizontal edge detection (Y-direction)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
        sobely_abs = np.uint8(np.clip(np.absolute(sobely), 0, 255))
        
        # Threshold and morphology
        _, thresh_y = cv2.threshold(sobely_abs, 50, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5), np.uint8)
        morph_y = cv2.morphologyEx(thresh_y, cv2.MORPH_CLOSE, kernel)
        
        # Get image size
        img_height, img_width = img.shape[:2]
        center_x = img_width // 2

        # Find contours to locate horizontal grout lines
        contours, _ = cv2.findContours(morph_y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find top grout line from bounding boxes
        top_candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h != 0 else 0
            y_center = y + h/2
            if aspect_ratio > 5 and (img_height * 0.15) < y_center < (img_height * 0.7):
                top_candidates.append((y_center, x, w))

        if top_candidates:
            # Sort by y-position
            top_candidates.sort(key=lambda tup: tup[0])
            avg_top_y, x_top, w_top = top_candidates[0]
            avg_top_y = int(avg_top_y)

            # Smarter side check
            if (x_top + w_top) < center_x:
                top_side = 'left'
            elif x_top > center_x:
                top_side = 'right'
            else:
                top_side = 'right'  # Default if straddling
        else:
            avg_top_y = int(img_height * 0.25)
            top_side = 'right'

        # Find bottom grout line using Sobel scanning
        crop_bottom_start = int(img_height * 0.8)
        crop_bottom_end = int(img_height * 0.95)
        bottom_band = gray[crop_bottom_start:crop_bottom_end, :]

        sobel_bottom = cv2.Sobel(bottom_band, cv2.CV_64F, 0, 1, ksize=5)
        sobel_bottom_profile = np.sum(np.abs(sobel_bottom), axis=1)

        peak_relative_y = np.argmax(sobel_bottom_profile)
        peak_absolute_y = crop_bottom_start + peak_relative_y

        # Determine side for bottom grout based on Sobel Y gradient strength
        band_y_range = 5
        left_band = np.sum(sobely_abs[peak_absolute_y-band_y_range:peak_absolute_y+band_y_range, :center_x])
        right_band = np.sum(sobely_abs[peak_absolute_y-band_y_range:peak_absolute_y+band_y_range, center_x:])
        bottom_side = 'left' if left_band > right_band else 'right'

        # Draw final image
        final_img = img.copy()

        # Draw top line
        if top_side == 'left':
            cv2.line(final_img, (center_x, avg_top_y), (0, avg_top_y), (0, 255, 255), 3)
        else:
            cv2.line(final_img, (center_x, avg_top_y), (img_width-1, avg_top_y), (0, 255, 255), 3)

        # Draw bottom line
        if bottom_side == 'left':
            cv2.line(final_img, (center_x, peak_absolute_y), (0, peak_absolute_y), (0, 255, 255), 3)
        else:
            cv2.line(final_img, (center_x, peak_absolute_y), (img_width-1, peak_absolute_y), (0, 255, 255), 3)      
        
        cv2.namedWindow("Processed Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Processed Image", 800, 600)
        cv2.imshow("Processed Image", final_img)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    grab_result.Release()

# Release the camera
camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()
