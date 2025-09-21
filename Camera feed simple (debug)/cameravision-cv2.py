from pypylon import pylon
import cv2
import numpy as np

# Initialize the camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Open the camera
camera.Open()

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
        cv2.imshow("Basler Camera Feed", img_array)

        # Save and print an image from the feed
        cv2.imwrite("frame.jpg", img_array)
        print("Saved frame.jpg")
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    grab_result.Release()

# Release the camera
camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()