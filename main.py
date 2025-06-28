import cv2 
from ultralytics import YOLO
from common_utils import crop_from_coords, extract_face_coords

face_det = YOLO("./models/yolov11m-face.pt")


# test_image = cv2.imread("/home/hbdesk/Downloads/webcam-toy-photo4.jpg")

# test_results = face_det(test_image)
# _a = extract_face_coords(test_results)
# crop_img = crop_from_coords(_a, test_image)


cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit() 

# Set camera resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize variables for features
show_edges = False
show_blur = False
show_gray = False
flip_horizontal = False

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    height, width, channels = frame.shape 
    if not ret:
        print("Error: Could not read frame")
        break
    cv2.imshow("Shitty Center Stage", cv2.flip(frame, 1))
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
