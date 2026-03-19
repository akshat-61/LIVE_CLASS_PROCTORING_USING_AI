from ultralytics import YOLO

MODEL_PATH = "best.pt"   # make sure this file exists in root

try:
    person_model = YOLO(MODEL_PATH)
    print("[INFO] YOLO model loaded successfully")
except Exception as e:
    print("[ERROR] Failed to load YOLO model:", e)
    person_model = None

AI_FRAME_SIZE = (640, 480)

CUSTOM_CONF_THRESHOLD = 0.25

CUSTOM_CLASS = {
    "PERSON": 0
}

SEAT_ZONE_RADIUS = 35

STABILITY_FRAMES = 10

EXAM_ID = None
ROOM_ID = None
ALERT_URL = "http://localhost:5000/alert"

def reset_session():
    global student_id_map, seat_positions

    try:
        student_id_map.clear()
    except:
        pass

    try:
        seat_positions.clear()
    except:
        pass

    print("[INFO] AI session reset")

# === DUMMY LEARNER (TEMP FIX) ===

class DummyLearner:
    def __init__(self):
        self.calibrated = {}

    def on_calibration_frame(self, *args, **kwargs):
        pass

    def on_lock(self, student_map):
        for sid in student_map.values():
            self.calibrated[sid] = True

    def set_seat_positions(self, positions):
        self.positions = positions

    def is_calibrated(self, sid):
        return self.calibrated.get(sid, False)


learner = DummyLearner()

# === POSE DETECTOR ===
import mediapipe as mp

try:
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("[INFO] Pose detector initialized")
except Exception as e:
    print("[ERROR] Pose detector failed:", e)
    pose_detector = None

# === STUDENT TRACKING SYSTEM ===

student_id_map = {}
seat_positions = {}

def assign_student_ids(positions):
    """
    Assigns student IDs like S1, S2, S3...
    """
    mapping = {}
    for i, tid in enumerate(positions.keys(), start=1):
        mapping[tid] = f"S{i}"
    return mapping

def init_seat_zones(positions):
    global seat_positions
    seat_positions = positions.copy()

# Placeholder internal states
locked = False
stable_counter = 0
_lock_frame = 0
_calib_accumulated_positions = {}

# === AI FRAME PROCESSING ===

import cv2

def run_ai_on_frame(frame):
    """
    Basic AI processing:
    - Detect persons using YOLO
    - Draw bounding boxes
    - Mark suspicious if too many people
    """

    if person_model is None:
        return frame

    results = person_model(frame, verbose=False)

    annotated = frame.copy()
    person_count = 0

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 0 and conf > 0.3:  # person class
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 🚨 Simple suspicious logic
    if person_count > 5:
        cv2.putText(
            annotated,
            "SUSPICIOUS: Too many people",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    return annotated

def shutdown():
    print("[INFO] Pose detector shut down")
