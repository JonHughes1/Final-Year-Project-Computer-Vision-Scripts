import cv2
from pypylon import pylon
import time
import socket
import os
import numpy as np
from ultralytics import YOLO

# Load YOLO model (replace with your own model path if needed)
model = YOLO("train_yolov8/weights/best.pt")

# Socket settings
PI_IP = "192.168.10.2"
PORT = 65432

def send_numbers(x, y):
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.connect((PI_IP, PORT))
		s.sendall(f"{x},{y}\n".encode())

def detect_line_params_and_annotate(img_array):
    offset, angle = 0.0, 0.0
    annotated_frame = img_array.copy()
    results = model.predict(img_array)
    height, width, _ = img_array.shape
    img_center_x = width // 2
    boxes = results[0].boxes
    best_vertical_box = None
    min_offset = float('inf')
    vertical_boxes = []
    horizontal_boxes = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        box_center_x = (x1 + x2) // 2
        box_center_y = (y1 + y2) // 2
        box_width = abs(x2 - x1)
        box_height = abs(y2 - y1)
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = results[0].names[cls]

        # Determine orientation
        if label.lower() == "v_line":
            # It's a vertical box — consider for center-most
            vertical_boxes.append((x1, y1, x2, y2, conf, label)) # Add to vertical line list
            offset = abs(box_center_x - img_center_x)
            if offset < min_offset:
                min_offset = offset
                best_vertical_box = (x1, y1, x2, y2, conf, label)
        else:
            # It's a horizontal box — draw it
            horizontal_boxes.append((x1, y1, x2, y2, conf, label)) # Add to horizontal line list
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(annotated_frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            # If we are on a junction
            if box_center_y > 2015 and box_center_y < 2030:
                send_numbers(7000,1)
                print("Junction")

        # Draw center-most vertical box (if found)
        if best_vertical_box:
            x1, y1, x2, y2, conf, label = best_vertical_box
            box_center_x = (x1 + x2) // 2
            offset_pixels = box_center_x - img_center_x

            # Draw bounding box and center line
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(annotated_frame, (img_center_x, 0), (img_center_x, height), (255, 0, 0), 1)

            # Annotate
            text = f'{label} {conf:.2f}'
            cv2.putText(annotated_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2) 
    #2015-2030
    
    return annotated_frame
    

# Initialize Basler camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.ExposureTime.SetValue(50000)
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_RGB8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# Frame loop
try:
    while camera.IsGrabbing():
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grab_result.GrabSucceeded():
        
            image = converter.Convert(grab_result)
            img_array = image.GetArray()
            
            # Save to temp file first to avoid reading from half written jpg
            tmp_filename = "Images/current_tmp.jpg"
            final_filename = "Images/current.jpg"
            cv2.imwrite(tmp_filename, img_array)
            os.rename(tmp_filename, final_filename) # atomic move

            # Detect + annotate
            annotated = detect_line_params_and_annotate(img_array)
            #print(f"[YOLO] Offset: {offset:.2f}, Angle: {angle:.2f}")
            
            # Check when we reach junction:
            # We do this by checking for the centrepoint of the horizontal line bounding boxes. If the centrepoint is within a certain range - detect junction
            # y = 2015 - 2030 ish

            
            cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLO Detection", 800, 600)
            cv2.imshow("YOLO Detection", annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        grab_result.Release()

finally:
    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()