import cv2
import torch
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import threading
from deep_sort_realtime.deepsort_tracker import DeepSort

class TrafficDetector:
    def __init__(self):
        self.CONFIDENCE_THRESHOLD = 0.6
        self.VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.RESIZE_DIM = (1020, 500)
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._initialize_model()
        self.tracker = DeepSort(max_age=30)
        self.directions = {}
        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.frame = None
        self.processed_frame = None
        self.running = True
        self.video_writer = None  # Video writer object

    def _initialize_model(self):
        model = YOLO('yolov8s.pt')
        if self.DEVICE == 'cuda':
            model = model.half()
        return model.to(self.DEVICE)

    def _process_frame(self, frame):
        results = self.model.predict(
            frame,
            conf=self.CONFIDENCE_THRESHOLD,
            classes=self.VEHICLE_CLASSES,
            verbose=False,
            device=self.DEVICE
        )
        boxes = results[0].boxes
        detections = []
        for box in boxes:
            cls_id = int(box.cls.item())
            if cls_id not in self.VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf.item())
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'vehicle'))

        tracks = self.tracker.update_tracks(detections, frame=frame)

        violations = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            cx = int((l + r) / 2)

            if track_id not in self.directions:
                self.directions[track_id] = [cx]
            else:
                self.directions[track_id].append(cx)
                if len(self.directions[track_id]) > 10:
                    self.directions[track_id].pop(0)

            direction_list = self.directions[track_id]
            if len(direction_list) >= 2 and direction_list[-1] - direction_list[0] > 20:
                violations.append(track_id)

            color = (0, 0, 255) if track_id in violations else (0, 255, 0)
            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), color, 2)
            cv2.putText(frame, f'ID {track_id}', (int(l), int(t) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if track_id in violations:
                cv2.putText(frame, 'WRONG WAY', (int(l), int(t) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def _worker(self):
        while self.running:
            if self.frame is not None:
                frame = self.frame.copy()
                processed_frame = self._process_frame(frame)
                self.processed_frame = processed_frame
                
                # Write the processed frame to video
                if self.video_writer is not None:
                    self.video_writer.write(processed_frame)

    def process_video(self, video_path, output_path='output.mp4'):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = self.RESIZE_DIM
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not self.video_writer.isOpened():
            raise IOError("Could not open the output video for writing")

        thread = threading.Thread(target=self._worker)
        thread.start()

        while True:
            ret, frame = cap.read()
            if not ret:
                self.running = False
                break

            frame = cv2.resize(frame, self.RESIZE_DIM)
            self.frame = frame

            if self.processed_frame is not None:
                cv2.imshow("Wrong-Way Detection", self.processed_frame)

            if cv2.waitKey(33) & 0xFF == ord('q'):
                self.running = False
                break

        thread.join()
        cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = TrafficDetector()
    input_video = 
    output_video = 
    detector.process_video(input_video, output_video)
