import cv2
import torch
import os
import numpy as np
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO
import threading
import math

class ParkingDetector:
    def __init__(self):
        self.MIN_PARKING_FRAMES = 800
        self.MOVEMENT_THRESHOLD = 12
        self.CONFIDENCE_THRESHOLD = 0.45
        self.VEHICLE_CLASSES = [2, 3, 5, 7]
        self.RESIZE_DIM = (1020, 500)
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = self._initialize_model()
        self.parked_vehicles = defaultdict(self._new_vehicle_dict)
        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        os.makedirs("wrong_parked_cars", exist_ok=True)

        self.frame = None
        self.processed_frame = None
        self.running = True
        self.video_writer = None  # Video writer object

    def _initialize_model(self):
        model = YOLO('yolov8s.pt')
        if self.DEVICE == 'cuda':
            model = model.half()
        return model.to(self.DEVICE)

    def _new_vehicle_dict(self):
        return {
            'center_history': [],
            'position': None,
            'saved': False,
            'movement': 0,
            'stationary_frames': 0
        }

    def _calculate_movement(self, current_center, history):
        if not history:
            return 0
        return np.linalg.norm(np.array(current_center) - np.array(history[0]))

    def _find_matching_vehicle(self, current_center, max_distance=20):
        closest_id, min_dist = None, float('inf')
        for vid, data in self.parked_vehicles.items():
            if data['center_history']:
                last_center = data['center_history'][-1]
                dist = np.linalg.norm(np.array(current_center) - np.array(last_center))
                if dist < min_dist and dist < max_distance:
                    min_dist = dist
                    closest_id = vid
        return closest_id

    def _update_vehicle_data(self, vehicle_id, current_center, bbox):
        vehicle = self.parked_vehicles[vehicle_id]
        vehicle['center_history'].append(current_center)
        vehicle['position'] = bbox
        vehicle['movement'] = self._calculate_movement(current_center, vehicle['center_history'])

        if len(vehicle['center_history']) > 1:
            last_dist = np.linalg.norm(
                np.array(current_center) - np.array(vehicle['center_history'][-2])
            )
            if last_dist < 3:
                vehicle['stationary_frames'] += 1
            else:
                vehicle['stationary_frames'] = max(0, vehicle['stationary_frames'] - 1)

    def _save_parked_vehicle(self, vehicle_id, frame):
        vehicle = self.parked_vehicles[vehicle_id]
        x1, y1, x2, y2 = vehicle['position']
        cropped = frame[y1:y2, x1:x2]
        filename = f"wrong_parked_cars/{vehicle_id}_{self.start_time}.png"
        cv2.imwrite(filename, cropped)
        vehicle['saved'] = True
        print(f"[INFO] Saved wrong parked car: {vehicle_id}")

    def _visualize_results(self, frame, boxes, parked_count):
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cls_id = int(box[5])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            vehicle_id = self._find_matching_vehicle((cx, cy))
            if vehicle_id and self.parked_vehicles[vehicle_id]['stationary_frames'] >= self.MIN_PARKING_FRAMES:
                color = (0, 0, 255)
                label = "WRONG PARKED"
                font_scale = 1.2
                font_thickness = 3

                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                text_x = x1 + (x2 - x1) // 2 - text_width // 2
                frame_number = int(datetime.now().timestamp() * 10)
                bounce_offset = int(5 * math.sin(frame_number * 0.3))
                text_y = max(y1 - 10 + bounce_offset, text_height + 5)

                cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
            else:
                color = (0, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(frame, f"Total Vehicles: {len(boxes)}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Wrong_Parked: {parked_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame

    def _process_frame(self, frame):
        results = self.model.predict(frame, conf=self.CONFIDENCE_THRESHOLD, classes=self.VEHICLE_CLASSES, verbose=False, device=self.DEVICE)
        boxes = results[0].boxes.data.cpu().numpy()
        return boxes

    def _worker(self):
        while self.running:
            if self.frame is not None:
                frame = self.frame.copy()
                boxes = self._process_frame(frame)
                parked_count = 0

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    cls_id = int(box[5])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    vehicle_id = self._find_matching_vehicle((cx, cy))
                    if vehicle_id is None:
                        vehicle_id = f"{cls_id}_{cx}_{cy}"
                        if vehicle_id not in self.parked_vehicles:
                            self.parked_vehicles[vehicle_id] = self._new_vehicle_dict()

                    self._update_vehicle_data(vehicle_id, (cx, cy), (x1, y1, x2, y2))

                    if self.parked_vehicles[vehicle_id]['stationary_frames'] >= self.MIN_PARKING_FRAMES:
                        parked_count += 1
                        if not self.parked_vehicles[vehicle_id]['saved']:
                            self._save_parked_vehicle(vehicle_id, frame)

                self.processed_frame = self._visualize_results(frame, boxes, parked_count)
                
                # Write the processed frame to video if writer is initialized
                if self.video_writer is not None and self.processed_frame is not None:
                    self.video_writer.write(self.processed_frame)

    def process_video(self, video_path, output_path='output.mp4'):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = self.RESIZE_DIM
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI format
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
                cv2.imshow("Wrong Parking Detection", self.processed_frame)

            if cv2.waitKey(33) & 0xFF == ord('q'):
                self.running = False
                break

        thread.join()
        cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ParkingDetector()
    input_video = r"C:\Users\IBRA\Desktop\New folder\parking.mp4"
    output_video = r"C:\Users\IBRA\Desktop\New folder\output_parking.mp4"
    detector.process_video(input_video, output_video)