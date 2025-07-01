
import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db
#from playsound import playsound  # For playing alarm sound

app = FastAPI()
# Initialize Firebase
cred = credentials.Certificate("")
firebase_admin.initialize_app(cred, {
    'databaseURL': ''
})

def update_firebase_status(road, status):
    ref = db.reference(f'roads/{road}')
    ref.update({'status': status})



class TrafficDetector:
    def __init__(self):
        self.CONFIDENCE_THRESHOLD = 0.45
        self.VEHICLE_CLASSES = [2, 3, 5, 7]
        self.RESIZE_DIM = (620, 360)
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO('yolov8n.pt').to(self.DEVICE) 
        self.last_switch_time = time.time()
        self.active_side = 'left'
        self.cap = cv2.VideoCapture("")
        self.frame_count = 0
        self.prev_boxes_left = []
        self.prev_boxes_right = []

    def _process_frame(self, frame):
        results = self.model.predict(
            frame,
            conf=self.CONFIDENCE_THRESHOLD,
            classes=self.VEHICLE_CLASSES,
            verbose=False,
            device=self.DEVICE
        )
        boxes = results[0].boxes.data.cpu().numpy()
        return boxes

    def draw_boxes(self, frame, boxes):
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def draw_traffic_light(self, frame, center_x, top_y, active_color):
        colors = {
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0),
            'off': (160, 160, 160)
        }
        r = 15
        s = 10
        ys = [top_y, top_y + 2 * (r + s), top_y + 4 * (r + s)]
        for i, color in zip(['red', 'yellow', 'green'], ys):
            col = colors[i] if i == active_color else colors['off']
            cv2.circle(frame, (center_x, color), r, col, -1)

    def _visualize_dual_results(self, frame, boxes_left, boxes_right):
        count_left = len(boxes_left)
        count_right = len(boxes_right)

        def get_status(c):
            if c < 6: return "Clear"
            elif c < 10: return "Normal"
            return "Jamming"

        status_left = get_status(count_left)
        status_right = get_status(count_right)
        update_firebase_status('road1', status_left)
        update_firebase_status('road2', status_right)
        if status_left == status_right:
            if time.time() - self.last_switch_time > 5:
                self.active_side = 'right' if self.active_side == 'left' else 'left'
                self.last_switch_time = time.time()
            signal_left = 'green' if self.active_side == 'left' else 'red'
            signal_right = 'green' if self.active_side == 'right' else 'red'
        elif count_left > count_right:
            signal_left, signal_right = 'green', 'red'
        else:
            signal_left, signal_right = 'red', 'green'

        h, w = frame.shape[:2]
        status_colors = {"Clear": (0, 255, 0), "Normal": (0, 255, 255), "Jamming": (0, 0, 255)}

        self.draw_traffic_light(frame, w // 4, 30, signal_left)
        cv2.putText(frame, status_left.upper(), (w // 4 - 60, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_colors[status_left], 2)

        self.draw_traffic_light(frame, 3 * w // 4, 30, signal_right)
        cv2.putText(frame, status_right.upper(), (3 * w // 4 - 60, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_colors[status_right], 2)

        return frame

    def get_processed_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.resize(frame, self.RESIZE_DIM)
        width = frame.shape[1]
        mid = width // 2

        left_road = cv2.resize(frame[:, :mid], (320, 240))
        right_road = cv2.resize(frame[:, mid:], (320, 240))

        self.frame_count += 1
        if self.frame_count % 4 == 0:
            boxes_left = self._process_frame(left_road)
            boxes_right = self._process_frame(right_road)
            self.prev_boxes_left = boxes_left
            self.prev_boxes_right = boxes_right
        else:
            boxes_left = self.prev_boxes_left
            boxes_right = self.prev_boxes_right

        self.draw_boxes(left_road, boxes_left)
        self.draw_boxes(right_road, boxes_right)

        frame[:, :mid] = cv2.resize(left_road, (mid, frame.shape[0]))
        frame[:, mid:] = cv2.resize(right_road, (mid, frame.shape[0]))

        return self._visualize_dual_results(frame, boxes_left, boxes_right)

detector = TrafficDetector()

def generate():
    while True:
        frame = detector.get_processed_frame()
        if frame is None:
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
         
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/")
def video_feed():
    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')