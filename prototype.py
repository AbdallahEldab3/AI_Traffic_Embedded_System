import cv2
import numpy as np
import serial
import time
import pandas as pd
from ultralytics import YOLO
import cvzone
import threading
import queue
import torch
import os
from datetime import datetime
from collections import OrderedDict
import firebase_admin
from firebase_admin import credentials, db
#from playsound import playsound  # For playing alarm sound


# Initialize Firebase
cred = credentials.Certificate()
firebase_admin.initialize_app(cred, {
    'databaseURL': ''
})

def update_firebase_status(road, status):
    ref = db.reference(f'roads/{road}')
    ref.update({'status': status})

# Configuration and Initialization
esp32 = serial.Serial('COM7', 115200, timeout=1)
CAR_CONFIDENCE_THRESHOLD = 0.5
ACCIDENT_CONFIDENCE_THRESHOLD = 0.7

model = YOLO("best.pt").to('cuda' if torch.cuda.is_available() else 'cpu')

with open(r"D:\Apps\123m\coco1.txt", "r") as file:
    class_list = file.read().splitlines()
video_source = ''
video = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
video_traffic = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)

for cap in [video, video_traffic]:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 5)

if not video.isOpened() or not video_traffic.isOpened():
    print("Error: Could not open video stream.")
    exit()

rect1_points = np.array([[360, 29], [523, 13], [594, 384], [310, 413]])
rect2_points = np.array([[668, 9], [839, 23], [900, 387], [622, 385]])

area1 = [(360, 29), (523, 13), (545, 124), (344, 148)]
area2 = [(336, 221), (557, 194), (594, 384), (310, 413)]
area3 = [(668, 9), (839, 23), (858, 135), (653, 124)]
area4 = [(649, 190), (869, 209), (900, 387), (622, 385)]

save_dir = "wrong_way_cars"
os.makedirs(save_dir, exist_ok=True)

crop_left = 1
crop_right = 1

green_level = "Unknown"
red_level = "Unknown"
last_green = "Unknown"
last_red = "Unknown"
traffic_lock = threading.Lock()
car_status = {}
wrong_way_cars = set()
accidents = set()
next_object_id = 1000

centroid_tracker = OrderedDict()
object_id_counter = 1000
max_disappeared = 50

ALARM_SOUND = "all.mp3"

def remove_white_areas(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    return cv2.bitwise_and(frame, frame, mask=mask)

def preprocess_frame(frame):
    return cv2.GaussianBlur(frame, (5, 5), 0)

def get_car_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 0, 50])
    upper_bound = np.array([180, 255, 255])
    return cv2.inRange(hsv, lower_bound, upper_bound)

def calculate_car_ratio(edges, car_mask, points):
    polygon_mask = np.zeros_like(car_mask)
    cv2.fillPoly(polygon_mask, [points], 255)
    masked_edges = cv2.bitwise_and(edges, edges, mask=polygon_mask)
    masked_car_area = cv2.bitwise_and(car_mask, car_mask, mask=polygon_mask)
    edge_pixels = cv2.countNonZero(masked_edges)
    car_area_pixels = cv2.countNonZero(masked_car_area)
    return (edge_pixels / car_area_pixels) * 250 if car_area_pixels else 0

def get_traffic_level(ratio):
    if ratio < 15:
        return "Clear"
    elif ratio < 30:
        return "Normal"
    else:
        return "Jamming"

def handle_esp32_commands(current_green, current_red):
    command_map = {
        ("Normal", "Clear"): 'z',
        ("Jamming", "Clear"): 'x',
        ("Clear", "Clear"): 'c',
        ("Clear", "Normal"): 'v',
        ("Jamming", "Normal"): 'b',
        ("Normal", "Normal"): 'n',
        ("Clear", "Jamming"): 'm',
        ("Normal", "Jamming"): 'a',
        ("Jamming", "Jamming"): 's',
    }
    command = command_map.get((current_green, current_red), '')
    if command:
        esp32.write(f"{command}\n".encode())
        esp32.flush()
        print(f"Sent command: {command} for {current_green}/{current_red}")

def show_mouse_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        global mouse_x, mouse_y
        mouse_x, mouse_y = x, y

mouse_x, mouse_y = -1, -1

def process_traffic_level():
    global green_level, red_level, last_green, last_red
    while True:
        for _ in range(30):
            video_traffic.grab()
        ret, frame = video_traffic.retrieve()
        if not ret:
            print("Failed to grab frame for traffic analysis.")
            continue

        height, width, _ = frame.shape
        cropped_frame = frame[:, crop_left:width - crop_right]

        filtered_frame = remove_white_areas(cropped_frame)
        preprocessed_frame = preprocess_frame(filtered_frame)
        edges = cv2.Canny(preprocessed_frame, 50, 150)
        car_mask = get_car_mask(filtered_frame)

        green_ratio = calculate_car_ratio(edges, car_mask, rect1_points)
        red_ratio = calculate_car_ratio(edges, car_mask, rect2_points)

        current_green_level = get_traffic_level(green_ratio)
        current_red_level = get_traffic_level(red_ratio)

        with traffic_lock:
            if (current_green_level != last_green) or (current_red_level != last_red):
                handle_esp32_commands(current_green_level, current_red_level)
                last_green = current_green_level
                last_red = current_red_level

            green_level = current_green_level
            red_level = current_red_level

            update_firebase_status('road1', current_green_level)
            update_firebase_status('road2', current_red_level)

        time.sleep(0.5)

traffic_thread = threading.Thread(target=process_traffic_level, daemon=True)
traffic_thread.start()

frame_queue = queue.Queue(maxsize=2)

def capture_frames():
    while True:
        ret, frame = video.read()
        if ret and not frame_queue.full():
            frame_queue.put(frame)

capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

cv2.namedWindow("Traffic Monitoring")
cv2.setMouseCallback("Traffic Monitoring", show_mouse_coordinates)


while True:
    if frame_queue.empty():
        continue

    frame = frame_queue.get()

    height, width, _ = frame.shape
    cropped_frame = frame[:, crop_left:width - crop_right]

    results = model(cropped_frame)
    detections = results[0].boxes.data.cpu().numpy()

    if detections.size > 0:
        px = pd.DataFrame(detections).astype("float")
        for _, row in px.iterrows():
            x1, y1, x2, y2, conf, class_id = map(float, row[:6])
            class_name = class_list[int(class_id)]

            if (class_name == "car" and conf < CAR_CONFIDENCE_THRESHOLD) or \
               (class_name == "accident" and conf < ACCIDENT_CONFIDENCE_THRESHOLD):
                continue

            x1, y1, x2, y2, class_id = map(int, [x1, y1, x2, y2, class_id])

            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2
            centroid = (centroid_x, centroid_y)

            object_id = None
            min_distance = float('inf')
            for obj_id, obj_centroid in centroid_tracker.items():
                distance = np.linalg.norm(np.array(centroid) - np.array(obj_centroid))
                if distance < 50 and distance < min_distance:
                    min_distance = distance
                    object_id = obj_id

            if object_id is None:
                object_id = object_id_counter
                object_id_counter += 1

            centroid_tracker[object_id] = centroid

            cv2.rectangle(cropped_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            if class_name == "car":
                text = f'{class_name}'  # Removed confidence score
                if object_id in car_status and car_status[object_id]["wrong_way"]:
                    text = "Wrong Way"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_x = x1 + (x2 - x1 - text_size[0]) // 2
                    text_y = y1 - 10
                    cv2.putText(cropped_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                    # Play alarm for wrong-way car
                
                else:
                    cvzone.putTextRect(cropped_frame, text, (x1, y1), 1, 1)

            if class_name == "car":
                cx = (x1 + x2) // 2
                cy = y2

                if object_id not in car_status:
                    car_status[object_id] = {
                        "int_area1": False,
                        "int_area2": False,
                        "int_area3": False,
                        "int_area4": False,
                        "wrong_way": False,
                        "saved": False,
                        "last_seen": time.time()
                    }

                int_area1 = cv2.pointPolygonTest(np.array(area1, np.int32), (cx, cy), False) >= 0
                int_area2 = cv2.pointPolygonTest(np.array(area2, np.int32), (cx, cy), False) >= 0
                int_area3 = cv2.pointPolygonTest(np.array(area3, np.int32), (cx, cy), False) >= 0
                int_area4 = cv2.pointPolygonTest(np.array(area4, np.int32), (cx, cy), False) >= 0

                print(f"Car {object_id}: area1={int_area1}, area2={int_area2}, area3={int_area3}, area4={int_area4}")

                car_status[object_id]["int_area1"] |= int_area1
                car_status[object_id]["int_area2"] |= int_area2
                car_status[object_id]["int_area3"] |= int_area3
                car_status[object_id]["int_area4"] |= int_area4
                car_status[object_id]["last_seen"] = time.time()

                if (car_status[object_id]["int_area2"] and car_status[object_id]["int_area1"]) or \
                   (car_status[object_id]["int_area4"] and car_status[object_id]["int_area3"]):
                    car_status[object_id]["wrong_way"] = True

                if car_status[object_id]["wrong_way"] and not car_status[object_id]["saved"]:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    car_image_path = os.path.join(save_dir, f'car_{object_id}_{timestamp}.png')
                    cv2.imwrite(car_image_path, cropped_frame[y1:y2, x1:x2])
                    car_status[object_id]["saved"] = True
                    wrong_way_cars.add(object_id)

            if class_name == "accident":
                accidents.add(object_id)
                cv2.rectangle(cropped_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                text = "Accident"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = x1 + (x2 - x1 - text_size[0]) // 2
                text_y = y1 - 10
                cv2.putText(cropped_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                # Play alarm for accident


    cv2.polylines(cropped_frame, [np.array(rect1_points, np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(cropped_frame, [np.array(rect2_points, np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(cropped_frame, [np.array(area1, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(cropped_frame, [np.array(area2, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(cropped_frame, [np.array(area3, np.int32)], True, (255, 0, 0), 2)
    cv2.polylines(cropped_frame, [np.array(area4, np.int32)], True, (255, 0, 0), 2)

    with traffic_lock:
        current_green = green_level
        current_red = red_level

    if mouse_x != -1 and mouse_y != -1:
        cv2.putText(cropped_frame, f"Mouse: ({mouse_x}, {mouse_y})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    # Change text color based on traffic level
    green_color = (0, 255, 0) if current_green == "Clear" else (0, 0, 255) if current_green == "Jamming" else (255, 0, 0)
    red_color = (0, 255, 0) if current_red == "Clear" else (0, 0, 255) if current_red == "Jamming" else (255, 0, 0)

    cv2.putText(cropped_frame, f" {current_green}", (140, 650), cv2.FONT_HERSHEY_SIMPLEX, 1.5, green_color, 3)
    cv2.putText(cropped_frame, f" {current_red}", (580, 650), cv2.FONT_HERSHEY_SIMPLEX, 1.5, red_color, 3)
    cv2.putText(cropped_frame, f'Wrong_Way_Cars: {len(wrong_way_cars)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Traffic Monitoring", cropped_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
video_traffic.release()
cv2.destroyAllWindows()
