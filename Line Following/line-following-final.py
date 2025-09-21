from pypylon import pylon
import cv2
import numpy as np
from skimage.morphology import skeletonize
import math
import socket
import time

PI_IP = "192.168.10.2"
PORT = 65432

def send_numbers(x, y):
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.connect((PI_IP, PORT))
		s.sendall(f"{x},{y}\n".encode())

# Grab frames
while (1):

        img_array = cv2.imread("Images/current.jpg")

        # Display the image
        #cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
        #cv2.resizeWindow("Camera Feed", 800, 600)
        #cv2.imshow("Camera Feed", img_array)
        
        # Do line follower image processing here:
        # Step 2: Edge detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # Remove noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Bilateral filter to further reduce noise
        blurred2 = cv2.bilateralFilter(blurred,15,25,25)

        # Detect edges using the Canny edge detector
        edges = cv2.Canny(blurred2, 50, 150, apertureSize = 5)
        
        # Perform skeletonization using skimage's skeletonize function
        skeleton = skeletonize(edges)

        # Convert back to 255 for display
        skeleton = (skeleton * 255).astype(np.uint8)
        
        # Remove small isolated components
        # Define the minimum pixel threshold for keeping components - this needs to change based on image size
        p = 250
        # Apply connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton, connectivity=8)

        # Create a mask to remove small components
        filtered_image = np.zeros_like(skeleton)

        # Iterate through all components except the background (index 0)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= p:  # Keep components larger than p pixels
                filtered_image[labels == i] = 255  # Set valid components to foreground (255)

        # Get image dimensions
        height, width = filtered_image.shape

        # Create a blank binary image (black background)
        blank_image = np.zeros((height, width), dtype=np.uint8)  # 1 channel for binary image

        # Apply Probabilistic Hough Line Transform
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)
        
        # Define thresholds for nearly vertical or horizontal lines
        vertical_threshold = 250  # Pixels (for filtering vertical lines)
        horizontal_threshold = 10  # Pixels (for filtering horizontal lines)
        edge_margin = 20  # Pixels (margin for filtering out lines near the edge)
        
        # Draw lines but exclude horizontal ones
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                dx = abs(x2 - x1)  # Horizontal difference (for vertical check)
                dy = abs(y2 - y1)  # Vertical difference (for horizontal check)

                # Exclude lines that are too close to the image edges
                if (x1 < edge_margin or x1 > width - edge_margin or x2 < edge_margin or x2 > width - edge_margin):
                    continue  # Skip this line

                # Keep only nearly vertical lines (filter out horizontal lines)
                if dx < vertical_threshold and dy > horizontal_threshold:
                    cv2.line(blank_image, (x1, y1), (x2, y2), 255, 2)  # Draw vertical lines in green

        # Output offset from centre and gradient dy/dx

        # Travel in from the edge at the bottom of the image to find the offset
        height, width = blank_image.shape
        list1 = []
        for i in range(width):
          if blank_image[(height-280), i] == 255:
            list1.append(i)
        
        # If the list has items:
        if list1:
            # Take smallest and largest point
            leftedge = min(list1)
            rightedge = max(list1)

            # Find centre of track
            centre = leftedge + rightedge
            centre = centre/2

            # Calculate offset
            truecentre = width/2
            offset = centre - truecentre

        # Go to somewhere up the line to calculate gradient (e.g. y=500)
        # Find centrepoint again
        height, width = blank_image.shape
        list2 = []
        for i in range(width):
          if blank_image[(height-900), i] == 255:
            list2.append(i)

        # If the list has items:
        if list2:
            # Take smallest and largest point
            leftedge1 = min(list2)
            rightedge1 = max(list2)

            # Find centre of track
            centre1 = leftedge1 + rightedge1
            centre1 = centre1/2

        if list1 and list2:
            # Calculate the gradient:
            point1 = (height-280,centre)
            point2 = (height-900,centre1)

            dy = height-50 - height-600
            dx = centre - centre1
            
            if dx != 0:
                gradient = dy/dx
                
            if dx == 0:
                gradient = 0

            anglerad = math.atan(gradient)
            angle = math.degrees(anglerad)

            print("gradient:", gradient)
            print("angle:", angle)
            print("offset:", offset)
        
            send_numbers(offset, angle)

        if not list1 or not list2:
            print("No line to follow")
            
        cv2.namedWindow("Processed Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Processed Image", 800, 600)
        cv2.imshow("Processed Image", blank_image)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
