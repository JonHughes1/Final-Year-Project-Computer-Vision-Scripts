Files:

1. Camera feed simple (debug)

contains scripts which just show live camera feed for debugging purposes, can be displayed using OpenCV or if the library is not installed matplotlib can be used.

 - cameravision-cv2
 - cameravision-matplotlib


2. Camera lens calibration script

contains a script to calibrate lens warping and counteract it to make line following more accurate, along with folder of checkerboard images used for calibration

 - checkerboard-old
 - lens-calibration-final


3. Line Following

contains main line following scripts

 - train_yolov8 - model training analytics from training & weights for trained model
 - line-following-final - contains non ML script used for detecting line edges and calculating offset and bearing, which are transmitted to the main 	controller module
 - non-ml-junction-detect-final - obsolete in the final implementation but a non-ML method of detecting the position of junctions (worse performance 	than YOLO model)
 - yolo-detection-final - YOLO model implementation used to detect the position of junctions, which sends a flag to the main controller when the 	platform needs to stop & turn.