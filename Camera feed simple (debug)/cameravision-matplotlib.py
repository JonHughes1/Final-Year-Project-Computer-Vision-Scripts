from pypylon import pylon
import numpy as np
import matplotlib.pyplot as plt

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
fig, ax = plt.subplots()
while camera.IsGrabbing():
    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grab_result.GrabSucceeded():
        # Convert the image to a NumPy array
        image = converter.Convert(grab_result)
        img_array = image.GetArray()

        # Display the image using Matplotlib
        ax.clear()
        ax.imshow(img_array)
        ax.set_title("Basler Camera Feed")
        plt.pause(0.01)  # Pause to update the figure
        
        # Save the frame
        plt.imsave("frame.jpg", img_array)
        print("Saved frame.jpg")
        
        # Check for user input to exit
        if plt.waitforbuttonpress(0.01):
            break

    grab_result.Release()

# Release the camera
camera.StopGrabbing()
camera.Close()
plt.close()
