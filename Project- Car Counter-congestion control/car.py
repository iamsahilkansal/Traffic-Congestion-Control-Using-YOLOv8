from ultralytics import YOLO
import cv2
import math
import cvzone

# Video source
cap = cv2.VideoCapture("../Videos/Test.mp4")

# Class names for YOLO detection
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Load YOLO model
model = YOLO("../Yolo-Weights/yolov8n.pt")

# Define line positions
line1_y = 200  # y-coordinate of first line
line2_y = 400  # y-coordinate of second line

# Custom x-coordinates for lines
line1_start_x, line1_end_x = 75, 500  # Inward Line Red
line2_start_x, line2_end_x = 475, 1275  # Outward Line Blue

# Tracking and counters
line1_tracked = set()  # Tracks vehicles crossing Line 1
line2_tracked = set()  # Tracks vehicles crossing Line 2
line1_counter = 0      # Counter for Line 1
line2_counter = 0      # Counter for Line 2

# ID dictionary for manual tracking
objects = {}
next_id = 0


def update_tracker(detections, objects, threshold=50):
    global next_id
    updated_objects = {}
    for det in detections:
        x1, y1, x2, y2 = det[:4]
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        matched_id = None
        for obj_id, (prev_x, prev_y) in objects.items():
            distance = math.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
            if distance < threshold:
                matched_id = obj_id
                break

        if matched_id is None:  # New object
            matched_id = next_id
            next_id += 1

        updated_objects[matched_id] = (center_x, center_y)

    return updated_objects


while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    detections = []


    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            bbox = (int(x1), int(y1), int(w), int(h))

            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            currClass = classNames[cls]

            # Process only vehicle classes
            if currClass in ["car", "truck", "bicycle", "motorbike", "bus"] and conf>0.1:
                cvzone.cornerRect(img, bbox, l=9)
                cvzone.putTextRect(img, f'{currClass} {conf}', (max(0, x1), max(35, y1)), thickness=1, scale=0.6,
                                   offset=5)
                detections.append([x1, y1, x2, y2, conf])  # Add detection for tracking

    # Update tracker with new detections
    objects = update_tracker(detections, objects)

    for obj_id, (center_x, center_y) in objects.items():
        cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), cv2.FILLED)

        # Count vehicles crossing Line 1
        if (line1_y - 10 < center_y < line1_y + 10) and (line1_start_x < center_x < line1_end_x) and obj_id not in line1_tracked:
            line1_counter += 1
            line1_tracked.add(obj_id)

        # Count vehicles crossing Line 2
        if (line2_y - 10 < center_y < line2_y + 10) and (line2_start_x < center_x < line2_end_x) and obj_id not in line2_tracked:
            line2_counter += 1
            line2_tracked.add(obj_id)

    # Drawing lines
    cv2.line(img, (line1_start_x, line1_y), (line1_end_x, line1_y), (0, 0, 255), 2)
    cv2.line(img, (line2_start_x, line2_y), (line2_end_x, line2_y), (255, 0, 0), 2)

    # Display counters
    cv2.putText(img, f'Inward Traffic Count: {line1_counter}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    cv2.putText(img, f'Outward Traffic Count: {line2_counter}', (800, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()