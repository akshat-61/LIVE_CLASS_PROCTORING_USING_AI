import cv2
import numpy as np
import time
import requests
import logging
import threading
import os
import torch
from ultralytics import YOLO
import mediapipe as mp
from core.adaptive_learning import AdaptiveLearner
from output.alert_logger import AlertLogger
from output.feature_collector import FeatureCollector
from detection.metrics_engine import MetricsEngine
from detection.anomaly_detector import AnomalyDetector
from model_loader import load_models
from classification.cheating_score import CheatingScore
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from collections import deque as _deque
from core.event_manager import EventManager
from tracking.student_state import StudentState
from output.evidence_manager import EvidenceManager
from detection.intelligence_layer import IntelligenceLayer
from classification.cheating_classifier import CheatingClassifier
from output.report_generator import ReportGenerator


log = logging.getLogger(__name__)
classifier = CheatingClassifier()
logger = AlertLogger()
collector = FeatureCollector()
metrics_engine = MetricsEngine(window_size=90)
anomaly_detector = AnomalyDetector()
score_engine = CheatingScore()
report_generator = ReportGenerator()
frame_queue = Queue(maxsize=120)
event_manager = EventManager()
student_state = StudentState()
evidence_manager = EvidenceManager()
intel = IntelligenceLayer()

_YOLO_WORKERS = 3
_MP_WORKERS   = 3
_yolo_pool    = ThreadPoolExecutor(max_workers=_YOLO_WORKERS, thread_name_prefix="yolo")
_mp_pool      = ThreadPoolExecutor(max_workers=_MP_WORKERS,   thread_name_prefix="mp")

_logical_cores    = os.cpu_count() or 4
_torch_threads    = max(1, min(4, _logical_cores // _YOLO_WORKERS))
torch.set_num_threads(_torch_threads)
log.info("torch threads per worker: %d  (logical cores: %d)", _torch_threads, _logical_cores)

os.makedirs("evidence", exist_ok=True)

_RAM_DISK      = "/dev/shm/proctoring_evidence"
_SSD_EVIDENCE  = "evidence"
_USE_RAM_DISK  = os.path.exists("/dev/shm")

if _USE_RAM_DISK:
    os.makedirs(_RAM_DISK, exist_ok=True)

_evidence_queue: Queue = Queue(maxsize=500)


def _evidence_writer_loop():
    import shutil

    while True:
        try:
            item = _evidence_queue.get(timeout=1.0)
        except Empty:
            continue

        if item is None:
            _evidence_queue.task_done()
            break

        try:
            op = item[0]

            if op == "write":
                _, folder, path, frame_data, quality = item
                os.makedirs(folder, exist_ok=True)
                cv2.imwrite(path, frame_data, [cv2.IMWRITE_JPEG_QUALITY, quality])

            elif op == "move":
                _, src_path, dst_folder = item
                os.makedirs(dst_folder, exist_ok=True)
                dst_path = os.path.join(dst_folder, os.path.basename(src_path))
                shutil.move(src_path, dst_path)

        except Exception as e:
            log.error("EvidenceWriter error: %s", e)
        finally:
            _evidence_queue.task_done()


_evidence_writer_thread = threading.Thread(
    target=_evidence_writer_loop,
    name="evidence-writer",
    daemon=True,
)
_evidence_writer_thread.start()

learner = AdaptiveLearner()
learner.load_baselines()

MODEL_DIR          = "/home/tx0978/Documents/classroom-proctoring/ai-engine/models/production"
BASE_DIR           = "/home/tx0978/Documents/classroom-proctoring/ai-engine"
PERSON_MODEL_PATH  = f"{MODEL_DIR}/person_model.pt"
GESTURE_MODEL_PATH = f"{MODEL_DIR}/gesture_model.pt"
OBJECT_MODEL_PATH  = f"{MODEL_DIR}/object_model.pt"
IDCARD_MODEL_PATH  = f"{MODEL_DIR}/id_model.pt"
COCO_MODEL_PATH    = "yolov8n.pt"

ALERT_URL    = "http://localhost:8080/api/proctoring/alert"
EXAM_ID      = "EXAM_001"
ROOM_ID      = "ROOM_A"

CUSTOM_CLASS = {"PERSON": 0}

OBJECT_CLASS_CONF = {
    "calculator": 0.55,
    "paper":      0.70,
    "person":     0.99,
}

INFERENCE_SKIP           = 2
GESTURE_CONF_THRESHOLD   = 0.55
PHONE_CLASS_ID           = 67
PHONE_MIN_AREA           = 120
PHONE_MAX_AREA           = 18000
PHONE_MAX_AREA_OCCUPIED  = 18000
PHONE_PROXIMITY_DIST     = 220
PHONE_PROXIMITY_MIN_DIST = 15
CUSTOM_CONF_THRESHOLD    = 0.45
COCO_CONF_THRESHOLD      = 0.45
STABILITY_FRAMES         = 20
ALERT_COOLDOWN_SEC       = 3
WHISPER_PROXIMITY_THRESHOLD = 80
MAR_VARIATION_THRESHOLD  = 0.004
HARD_MAR_THRESHOLD       = 0.18
MAR_TALK_FRAMES          = 3
MAR_WINDOW_SIZE          = 5
TALK_COUNTER_INCREMENT   = 2.0
TALK_COUNTER_DECAY       = 1.5
LOOK_DOWN_THRESHOLD      = 0.035
SIDE_GAZE_RATIO          = 2.8
LEAN_THRESHOLD           = 0.8
LEAN_CONFIRM_FRAMES      = 6
FACE_MATCH_RADIUS        = 120
POSE_FRAME_MOD           = 0
MESH_FRAME_MOD           = 1
HAND_FRAME_MOD           = 2
HAND_MOUTH_DIST          = 80
HAND_FACE_DIST           = 120
PALM_WRITE_DIST          = 60
WRITE_MOTION_FRAMES      = 3
HAND_CONFIRM_FRAMES      = 1
PASS_DIST_THRESHOLD      = 75
SIGNAL_CONFIRM_FRAMES    = 3
SEAT_ZONE_RADIUS         = 28
SEAT_VACANCY_FRAMES      = 450
VACANCY_GRACE_FRAMES     = 45
MAX_VACANCY_DIST         = 100
POST_LOCK_SETTLE_FRAMES  = 10
PROXIMITY_ONLY_DIST      = 40
PROXIMITY_CONFIRM_FRAMES = 12
SEAT_VACANCY_COOLDOWN    = 5.0
MIN_SEAT_PAIR_DIST       = 80
GAZE_RIGHT_THRESHOLD     = 0.58

CALC_PERSIST_WINDOW      = 60
CALC_PERSIST_RATE        = 0.20
CALC_PERSIST_COOLDOWN    = 5.0

SEAT_ASSIGN_MARGIN       = 35
SEAT_ASSIGN_ABS_CAP      = 65
PHONE_CONF_THRESHOLD = 0.45
GAZE_LEFT_THRESHOLD  = 0.42
AI_FRAME_SIZE        = (960, 540)

INVIGILATOR_HEIGHT_RATIO = 1.35
_median_student_bbox_h: float = 0.0
_invigilator_tids: set = set()

FRAME_MOD_BASE    = 1
GESTURE_FRAME_MOD = 2
YOLO_FRAME_MOD    = 2

FACE_ABSENT_FRAMES_THRESHOLD = 15
IDCARD_CONF_THRESHOLD        = 0.72

OBJECT_LABEL_EXCLUSIONS = {"hand", "arm", "fist", "pen", "open_palm", "person"}

GESTURE_FACE_OVERLAP_RADIUS = 80
WRIST_BELOW_FACE_MIN_PX     = 20

_lean_smooth: dict = {}

log.info("=" * 55)
log.info("  AI Proctoring System — Loading Models")
log.info("=" * 55)

person_model, gesture_model, object_model, idcard_model = load_models()

coco_model = YOLO(COCO_MODEL_PATH)
log.info("COCO model loaded.")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=16,
    refine_landmarks=True,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2,
)

mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4,
)

mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(
    max_num_hands=6,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.4,
)
log.info("MediaPipe loaded.")
log.info("=" * 55)

last_custom_results   = None
last_coco_results     = None
locked                = False
stable_counter        = 0
student_id_map        = {}
last_alert_time       = {}
mouth_counters        = {}
mar_history           = {}
gaze_history          = {}
head_ratio_history    = {}
frame_count           = 0
last_face_landmarks   = []
last_talking_students = set()
hand_mouth_counters   = {}
hand_face_counters    = {}
write_palm_counters   = {}
write_tip_history     = {}
_wrist_prev_positions: dict = {}
signaling_counters    = {}
phone_counters        = {}
seat_positions        = {}
seat_vacancy_counters = {}
seat_grace_counters   = {}
seat_was_vacant       = {}
proximity_counters    = {}
seat_initial_pair_dists = {}
id_visible_counters   = {}
last_face_landmarks_frame = -999
face_based_positions: dict = {}
_calib_accumulated_positions: dict = {}
_lock_frame: int = -999
_SETTLE_FRAMES: int = 2

_prev_gray: np.ndarray | None = None
_motion_burst_counter: int    = 0

calc_seen_window: dict = {}


def reset_session():
    global locked, stable_counter, student_id_map, frame_count
    global last_custom_results, last_coco_results
    global last_face_landmarks, last_talking_students, last_face_landmarks_frame
    global face_based_positions
    global _calib_accumulated_positions, _lock_frame
    global last_alert_time, mouth_counters, mar_history, gaze_history
    global hand_mouth_counters, hand_face_counters, write_palm_counters
    global write_tip_history, signaling_counters, phone_counters
    global seat_positions, seat_vacancy_counters, seat_grace_counters, \
           seat_was_vacant
    global proximity_counters, seat_initial_pair_dists
    global id_visible_counters
    global _prev_gray, _motion_burst_counter
    global calc_seen_window
    global _median_student_bbox_h, _invigilator_tids
    global intel

    locked                       = False
    stable_counter               = 0
    student_id_map               = {}
    frame_count                  = 0
    last_custom_results          = None
    last_coco_results            = None
    last_face_landmarks          = []
    last_talking_students        = set()
    last_face_landmarks_frame    = -999
    face_based_positions         = {}
    _calib_accumulated_positions = {}
    _lock_frame                  = -999
    last_alert_time              = {}
    mouth_counters               = {}
    mar_history                  = {}
    gaze_history                 = {}
    hand_mouth_counters          = {}
    hand_face_counters           = {}
    write_palm_counters          = {}
    write_tip_history            = {}
    _wrist_prev_positions        = {}
    signaling_counters           = {}
    phone_counters               = {}
    seat_positions               = {}
    seat_vacancy_counters        = {}
    seat_grace_counters          = {}
    seat_was_vacant              = {}
    proximity_counters           = {}
    seat_initial_pair_dists      = {}
    id_visible_counters          = {}
    _prev_gray                   = None
    _motion_burst_counter        = 0
    calc_seen_window             = {}
    _median_student_bbox_h       = 0.0
    _invigilator_tids            = set()
    event_manager.reset()
    student_state.reset()
    evidence_manager.reset()
    intel.reset()
    log.info("Session reset complete.")


def save_evidence(frame, student_id, event):
    timestamp  = time.strftime("%H-%M-%S")
    ssd_folder = f"{_SSD_EVIDENCE}/{event}"
    filename   = f"{student_id}_{timestamp}.jpg"
    ssd_path   = f"{ssd_folder}/{filename}"
    frame_copy = frame.copy()

    if _USE_RAM_DISK:
        ram_folder = f"{_RAM_DISK}/{event}"
        ram_path   = f"{ram_folder}/{filename}"
        try:
            _evidence_queue.put_nowait(("write", ram_folder, ram_path, frame_copy, 85))
            _evidence_queue.put_nowait(("move",  ram_path,   ssd_folder))
        except Exception:
            try:
                _evidence_queue.put_nowait(("write", ssd_folder, ssd_path, frame_copy, 85))
            except Exception:
                log.warning("EvidenceWriter queue full — dropped %s (%s)", event, student_id)
    else:
        try:
            _evidence_queue.put_nowait(("write", ssd_folder, ssd_path, frame_copy, 85))
        except Exception:
            log.warning("EvidenceWriter queue full — dropped %s (%s)", event, student_id)

    log.info("Evidence queued → %s", ssd_path)
    return ssd_path

def _now():
    return time.time()

def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def can_send(key, cooldown=None):
    cd = cooldown if cooldown is not None else ALERT_COOLDOWN_SEC
    frames_required = int(cd * 30.0)
    last_frame = last_alert_time.get(key, -999999)
    if (frame_count - last_frame) > frames_required:
        last_alert_time[key] = frame_count
        return True
    return False

def send_event_async(event_type, student_id=None, distance=None):
    def task():
        payload = {
            "examId": EXAM_ID,
            "roomId": ROOM_ID,
            "events": [{
                "type": event_type,
                "timestamp": _now(),
                "confidence": 0.9,
                "trackId": student_id,
                "distance": distance,
            }],
        }
        log.info("ALERT → %s | Student: %s | %s", event_type, student_id, time.strftime('%H:%M:%S'))
        try:
            requests.post(ALERT_URL, json=payload, timeout=1)
        except Exception:
            pass
    threading.Thread(target=task, daemon=True).start()

def assign_student_ids(positions):
    sorted_list = sorted(positions.items(), key=lambda item: (item[1][1], item[1][0]))
    return {tid: f"S{i:03d}" for i, (tid, _) in enumerate(sorted_list, 1)}

def get_student_at(x, y, current_positions):
    min_d, found = FACE_MATCH_RADIUS, None
    for tid, pos in current_positions.items():
        d = dist((x, y), pos)
        if d < min_d:
            min_d, found = d, tid
    return found


def get_seat_at(x, y):
    if not seat_positions:
        return None

    ranked = sorted(
        ((dist((x, y), seat_pos), sid) for sid, seat_pos in seat_positions.items())
    )

    if not ranked:
        return None

    best_d, best_sid = ranked[0]

    if best_d > SEAT_ASSIGN_ABS_CAP:
        return None

    if len(ranked) >= 2:
        second_d = ranked[1][0]
        if second_d - best_d < SEAT_ASSIGN_MARGIN:
            return None

    return best_sid


def prune_state(current_ids):
    active = set(student_id_map.get(tid, "") for tid in current_ids)
    for d in [
        mouth_counters, mar_history,
        hand_mouth_counters, hand_face_counters,
        write_palm_counters, phone_counters,
        write_tip_history, signaling_counters,
    ]:
        for k in list(d.keys()):
            if k not in active:
                del d[k]
    for k in list(_lean_smooth.keys()):
        if k not in active:
            del _lean_smooth[k]

def mouth_aspect_ratio(landmarks):
    def pt(i):
        lm = landmarks[i]; return (lm.x, lm.y)
    v1 = dist(pt(13), pt(14))
    v2 = dist(pt(312), pt(317))
    return ((v1 + v2) / 2.0) / (dist(pt(61), pt(291)) + 1e-6)

def is_mouth_open_vertically(landmarks, threshold=0.014):
    def pt(i):
        lm = landmarks[i]; return (lm.x, lm.y)
    v1 = dist(pt(13), pt(14))
    v2 = dist(pt(82), pt(312))
    return (v1 + v2) / 2 > threshold

def get_look_down_score(landmarks):
    def pt(i):
        lm = landmarks[i]; return (lm.x, lm.y)
    li = pt(468); ri = pt(473)
    ley = (pt(159)[1] + pt(145)[1]) / 2
    rey = (pt(386)[1] + pt(374)[1]) / 2
    return ((li[1] - ley) + (ri[1] - rey)) / 2

def get_gaze_direction(landmarks):
    left_iris_x  = landmarks[468].x
    right_iris_x = landmarks[473].x
    left_outer   = landmarks[33].x
    left_inner   = landmarks[133].x
    right_outer  = landmarks[263].x
    right_inner  = landmarks[362].x
    left_ratio  = (left_iris_x  - left_outer)  / (left_inner  - left_outer  + 1e-6)
    right_ratio = (right_iris_x - right_inner) / (right_outer - right_inner + 1e-6)
    iris_gaze = (left_ratio + right_ratio) / 2
    nose_x    = landmarks[1].x
    head_ratio = (nose_x - left_outer) / (right_outer - left_outer + 1e-6)
    looking_left  = (iris_gaze  < GAZE_LEFT_THRESHOLD)  or (head_ratio < 0.35)
    looking_right = (iris_gaze  > GAZE_RIGHT_THRESHOLD) or (head_ratio > 0.65)
    return iris_gaze, looking_left, looking_right


def draw_alert_banner(frame, text, color=(0, 130, 0)):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, (10, 33), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)

def get_palm_center(hand_lms, iw, ih):
    w = hand_lms.landmark[0]; m = hand_lms.landmark[9]
    return (int(((w.x + m.x) / 2) * iw), int(((w.y + m.y) / 2) * ih))

def get_index_tip(hand_lms, iw, ih):
    lm = hand_lms.landmark[8]
    return (int(lm.x * iw), int(lm.y * ih))

def get_wrist(hand_lms, iw, ih):
    lm = hand_lms.landmark[0]
    return (int(lm.x * iw), int(lm.y * ih))

def get_face_center(face_landmarks, iw, ih):
    lm = face_landmarks.landmark[1]
    return (int(lm.x * iw), int(lm.y * ih))

def get_mouth_center(face_landmarks, iw, ih):
    u = face_landmarks.landmark[13]; l = face_landmarks.landmark[14]
    return (int(((u.x + l.x) / 2) * iw), int(((u.y + l.y) / 2) * ih))

def get_all_face_centers(face_landmarks_list, iw, ih):
    return [get_face_center(f, iw, ih) for f in (face_landmarks_list or [])]

def point_is_on_face(point, face_centers, radius=GESTURE_FACE_OVERLAP_RADIUS):
    return any(dist(point, fc) < radius for fc in face_centers)

def init_seat_zones(positions):
    global seat_positions, seat_vacancy_counters, seat_grace_counters, \
           proximity_counters, seat_initial_pair_dists, seat_was_vacant
    seat_positions = {}
    for tid, pos in positions.items():
        sid = student_id_map.get(tid, "S???")
        seat_positions[sid] = pos

    DEDUP_THRESHOLD = 10
    dedup_sids = list(seat_positions.keys())
    removed = set()
    for i in range(len(dedup_sids)):
        if dedup_sids[i] in removed:
            continue
        for j in range(i + 1, len(dedup_sids)):
            if dedup_sids[j] in removed:
                continue
            d = dist(seat_positions[dedup_sids[i]], seat_positions[dedup_sids[j]])
            if d < DEDUP_THRESHOLD:
                log.warning("SeatDedup: %s is %.1fpx from %s — merging into %s",
                            dedup_sids[j], d, dedup_sids[i], dedup_sids[i])
                removed.add(dedup_sids[j])
    for sid in removed:
        del seat_positions[sid]

    seat_vacancy_counters = {
        sid: 0 for sid in seat_positions if sid in student_id_map.values()
    }
    seat_grace_counters = {sid: 0 for sid in seat_positions}
    seat_was_vacant     = {sid: False for sid in seat_positions}
    proximity_counters      = {}
    seat_ids                = list(seat_positions.keys())
    seat_initial_pair_dists = {}
    for i in range(len(seat_ids)):
        for j in range(i + 1, len(seat_ids)):
            a, b = seat_ids[i], seat_ids[j]
            key  = (min(a, b), max(a, b))
            seat_initial_pair_dists[key] = dist(seat_positions[a], seat_positions[b])
            proximity_counters[key]      = 0
    log.info("Seat zones locked: %s", seat_positions)

def assign_seats_voronoi(current_positions):
    detection_to_seat = {}
    for tid, pos in current_positions.items():
        best_sid, best_d = None, float("inf")
        for sid, seat_pos in seat_positions.items():
            d = dist(pos, seat_pos)
            if d < best_d:
                best_d, best_sid = d, sid
        if best_sid is not None:
            detection_to_seat[tid] = (best_sid, best_d)

    seat_to_best = {}
    for tid, (sid, d) in detection_to_seat.items():
        if sid not in seat_to_best or d < seat_to_best[sid][1]:
            seat_to_best[sid] = (tid, d)

    occupancy = {}
    for sid in seat_positions:
        if sid in seat_to_best and seat_to_best[sid][1] <= MAX_VACANCY_DIST:
            occupancy[sid] = seat_to_best[sid][0]
        else:
            occupancy[sid] = None
    return occupancy


def find_occupant(seat_pos, current_positions):
    best_tid, best_d = None, MAX_VACANCY_DIST
    for tid, pos in current_positions.items():
        d = dist(seat_pos, pos)
        if d < best_d:
            best_d, best_tid = d, tid
    return best_tid


def draw_seat_zones(frame):
    for sid, pos in seat_positions.items():
        overlay = frame.copy()
        cv2.circle(overlay, pos, SEAT_ZONE_RADIUS, (255, 200, 0), 2)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(frame, sid,
                    (pos[0] - 20, pos[1] + SEAT_ZONE_RADIUS + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)


def check_seat_zones(current_positions, frame):
    alert = None

    occupancy = assign_seats_voronoi(current_positions)

    for sid, seat_pos in seat_positions.items():
        occupant = occupancy.get(sid)

        if occupant is not None:
            seat_grace_counters[sid]   = 0
            seat_vacancy_counters[sid] = 0
            if seat_was_vacant.get(sid, False):
                seat_was_vacant[sid] = False
                if can_send(f"returned_{sid}", cooldown=SEAT_VACANCY_COOLDOWN):
                    logger.log_event(
                        student_id=sid,
                        event="STUDENT_RETURNED",
                        confidence=1.0,
                        frame=frame_count
                    )
                    send_event_async("STUDENT_RETURNED", sid)
                    score = score_engine.add_event(sid, "STUDENT_RETURNED")
                    if score > 80:
                        log.warning("CHEATING HIGH RISK: %s | score=%s", sid, score)
                alert = f"STUDENT RETURNED: {sid}"
                cv2.circle(frame, seat_pos, SEAT_ZONE_RADIUS, (0, 255, 0), 2)
        else:
            post_lock_settled = (frame_count - _lock_frame) > POST_LOCK_SETTLE_FRAMES

            if not post_lock_settled:
                seat_grace_counters[sid]   = 0
                seat_vacancy_counters[sid] = 0
            else:
                seat_grace_counters[sid] = seat_grace_counters.get(sid, 0) + 1

                if seat_grace_counters[sid] >= VACANCY_GRACE_FRAMES:
                    if seat_vacancy_counters.get(sid, 0) < SEAT_VACANCY_FRAMES:
                        seat_vacancy_counters[sid] = seat_vacancy_counters.get(sid, 0) + 1

                    if seat_vacancy_counters[sid] >= SEAT_VACANCY_FRAMES:
                        if not seat_was_vacant.get(sid, False):
                            seat_was_vacant[sid] = True
                            known_sid = (sid != "S???" and sid in set(student_id_map.values()))
                            if known_sid and can_send(f"vacancy_{sid}", cooldown=SEAT_VACANCY_COOLDOWN):
                                logger.log_event(
                                    student_id=sid,
                                    event="SEAT_VACATED",
                                    confidence=1.0,
                                    frame=frame_count
                                )
                                send_event_async("SEAT_VACATED", sid)
                                score = score_engine.add_event(sid, "SEAT_VACATED")
                                if score > 80:
                                    log.warning("CHEATING HIGH RISK: %s | score=%s", sid, score)
                            if known_sid:
                                alert = f"SEAT VACATED: {sid}"

            cv2.circle(frame, seat_pos, SEAT_ZONE_RADIUS, (0, 0, 200), 2)

    seat_ids = list(seat_positions.keys())
    for i in range(len(seat_ids)):
        for j in range(i + 1, len(seat_ids)):
            sid_a, sid_b = seat_ids[i], seat_ids[j]
            pair_key = (min(sid_a, sid_b), max(sid_a, sid_b))
            if seat_initial_pair_dists.get(pair_key, 9999) < MIN_SEAT_PAIR_DIST:
                continue
            occ_a = occupancy.get(sid_a)
            occ_b = occupancy.get(sid_b)
            if occ_a is None or occ_b is None:
                proximity_counters[pair_key] = 0
                continue
            d = dist(current_positions[occ_a], current_positions[occ_b])
            if d < PROXIMITY_ONLY_DIST:
                proximity_counters[pair_key] = proximity_counters.get(pair_key, 0) + 1
                if proximity_counters[pair_key] >= PROXIMITY_CONFIRM_FRAMES:
                    close_key = f"prox_{pair_key[0]}_{pair_key[1]}"
                    if can_send(close_key, cooldown=15.0):
                        logger.log_event(
                            student_id=f"{sid_a}-{sid_b}",
                            event="STUDENTS_TOO_CLOSE",
                            confidence=0.9,
                            frame=frame_count
                        )
                        send_event_async("STUDENTS_TOO_CLOSE", f"{sid_a}-{sid_b}", distance=int(d))
                        score = score_engine.add_event(sid_a, "STUDENTS_TOO_CLOSE")
                        if score > 80:
                            log.warning("CHEATING HIGH RISK: %s | score=%s", sid_a, score)
                    alert = f"STUDENTS TOO CLOSE: {sid_a} & {sid_b}"
                    learner.escalate_if_coordinated(sid_a, sid_b)
            else:
                proximity_counters[pair_key] = max(0, proximity_counters.get(pair_key, 0) - 1)
    return alert

def process_hands(hand_res, face_landmarks_list, current_positions, iw, ih, frame, talking_students):
    current_alert = None
    if not hand_res or not hand_res.multi_hand_landmarks:
        return current_alert
    fired_this_frame: set = set()

    face_centers = get_all_face_centers(face_landmarks_list, iw, ih)

    hands_data = []
    for hand_lms in hand_res.multi_hand_landmarks:
        palm  = get_palm_center(hand_lms, iw, ih)
        tip   = get_index_tip(hand_lms, iw, ih)
        wrist = get_wrist(hand_lms, iw, ih)

        if face_centers:
            nearest_face = min(face_centers, key=lambda fc: dist(wrist, fc))
            nearest_face_dist = dist(wrist, nearest_face)
            wrist_below = wrist[1] > nearest_face[1] + WRIST_BELOW_FACE_MIN_PX
            if nearest_face_dist < 70 or not wrist_below:
                continue

        tid = get_student_at(wrist[0], wrist[1], current_positions)
        if tid is None:
            tid = get_student_at(palm[0], palm[1], current_positions)
        sid = student_id_map.get(tid, None)
        if sid is None:
            continue
        hands_data.append({"palm": palm, "tip": tip, "wrist": wrist, "sid": sid})
        mp.solutions.drawing_utils.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        mouth_pos = None
        for f_lms in face_landmarks_list or []:
            fc = get_face_center(f_lms, iw, ih)
            if tid and dist(fc, current_positions.get(tid, (9999, 9999))) < FACE_MATCH_RADIUS:
                mouth_pos = get_mouth_center(f_lms, iw, ih)
                break

        if mouth_pos is not None:
            d_wrist = dist(wrist, mouth_pos)
            if d_wrist < 220:
                hand_face_counters[sid] = hand_face_counters.get(sid, 0) + 1
                cv2.circle(frame, wrist, 18, (0, 0, 255), 2)
            else:
                hand_face_counters[sid] = max(0, hand_face_counters.get(sid, 0) - 1)
            if hand_face_counters.get(sid, 0) >= HAND_CONFIRM_FRAMES:
                is_talking = mouth_counters.get(sid, 0) >= MAR_TALK_FRAMES
                event = "HANDS_ON_FACE"
                key   = f"handface_{sid}" if is_talking else f"faceonmouth_{sid}"
                if can_send(key):
                    ev = event_manager.emit(sid, "HANDS_ON_FACE", 0.9)
                    student_state.add_event(sid, ev)
                    logger.log_event(
                        student_id=sid,
                        event="HANDS_ON_FACE",
                        confidence=0.9,
                        frame=frame_count
                    )
                    send_event_async(event, sid)
                    score = score_engine.add_event(sid, "HANDS_ON_FACE")
                    if score > 80:
                        log.warning("CHEATING HIGH RISK: %s | score=%s", sid, score)
                if current_alert is None:
                    current_alert = f"HANDS ON FACE: {sid}"

    for i, h1 in enumerate(hands_data):
        for j, h2 in enumerate(hands_data):
            if i >= j or h1["sid"] != h2["sid"]:
                continue
            sid = h1["sid"]
            if dist(h1["wrist"], h2["wrist"]) < 160:
                last_pos    = write_tip_history.get(sid)
                current_pos = h1["tip"]
                motion      = dist(current_pos, last_pos) if last_pos else 0
                write_tip_history[sid] = current_pos
                if 2 < motion < 20:
                    write_palm_counters[sid] = write_palm_counters.get(sid, 0) + 1
                else:
                    write_palm_counters[sid] = max(0, write_palm_counters.get(sid, 0) - 1)
                if write_palm_counters.get(sid, 0) >= 6:
                    if can_send(f"writepalm_{sid}", cooldown=15.0):
                        ev = event_manager.emit(sid, "WRITING_ON_PALM", 0.9)
                        student_state.add_event(sid, ev)
                        logger.log_event(
                            student_id=sid,
                            event="WRITING_ON_PALM",
                            confidence=0.9,
                            frame=frame_count
                        )
                        send_event_async("WRITING_ON_PALM", sid)
                        score = score_engine.add_event(sid, "WRITING_ON_PALM")
                        if score > 80:
                            log.warning("CHEATING HIGH RISK: %s | score=%s", sid, score)
                    current_alert = f"WRITING ON PALM: {sid}"

    for i, h1 in enumerate(hands_data):
        for j, h2 in enumerate(hands_data):
            if i >= j or h1["sid"] == h2["sid"]:
                continue
            sid1, sid2 = h1["sid"], h2["sid"]
            if dist(h1["wrist"], h2["wrist"]) < PASS_DIST_THRESHOLD:
                cv2.line(frame, h1["wrist"], h2["wrist"], (0, 0, 255), 2)
                pair_key = f"pass_{min(sid1, sid2)}_{max(sid1, sid2)}"
                if pair_key in fired_this_frame:
                    continue

                from collections import deque as _dq
                for _sid, _wrist in [(sid1, h1["wrist"]), (sid2, h2["wrist"])]:
                    if _sid not in _wrist_prev_positions:
                        _wrist_prev_positions[_sid] = _dq(maxlen=5)
                    _wrist_prev_positions[_sid].append(_wrist)

                _wrists_moving = True
                hist1 = _wrist_prev_positions.get(sid1)
                hist2 = _wrist_prev_positions.get(sid2)
                if hist1 and hist2 and len(hist1) >= 3 and len(hist2) >= 3:
                    move1 = dist(hist1[0], hist1[-1])
                    move2 = dist(hist2[0], hist2[-1])
                    _wrists_moving = (move1 > 8 or move2 > 8)

                if not _wrists_moving:
                    continue

                if can_send(pair_key, cooldown=30.0):
                    fired_this_frame.add(pair_key)
                    save_evidence(frame, sid1, "PASSING_OBJECT")
                    learner.register_event(sid1, "PASSING_OBJECT")
                    learner.register_event(sid2, "PASSING_OBJECT")
                    ev = event_manager.emit(f"{sid1}-{sid2}", "PASSING_OBJECT", 0.9)
                    student_state.add_event(sid1, ev)
                    logger.log_event(
                        student_id=f"{sid1}-{sid2}",
                        event="PASSING_OBJECT",
                        confidence=0.9,
                        frame=frame_count
                    )
                    send_event_async("PASSING_OBJECT", f"{sid1}-{sid2}")
                    score = score_engine.add_event(sid1, "PASSING_OBJECT")
                    if score > 80:
                        log.warning("CHEATING HIGH RISK: %s | score=%s", sid1, score)
                current_alert = f"PASSING OBJECT: {sid1} & {sid2}"

    return current_alert

_fps_last_time: float = time.time()
_fps_value: float     = 0.0

def run_ai_on_frame(frame: np.ndarray) -> np.ndarray:
    global locked, stable_counter, student_id_map, frame_count
    global last_custom_results, last_coco_results
    global last_face_landmarks, last_talking_students, last_face_landmarks_frame
    global face_based_positions
    global _fps_last_time, _fps_value
    global _lock_frame
    global _median_student_bbox_h, _invigilator_tids

    frame_count += 1
    frame = cv2.resize(frame, AI_FRAME_SIZE)

    if locked and _lock_frame == -999:
        _lock_frame = frame_count
        log.info("_lock_frame anchored to frame %d (batch pre-lock detected)", frame_count)

    ih, iw        = frame.shape[:2]
    img_rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    current_alert = "All Clear"
    current_positions      = {}
    phones                 = []
    current_ids            = set()
    id_detected_this_frame = set()
    gesture_results        = None

    now = time.time()
    elapsed = now - _fps_last_time
    if elapsed > 0:
        _fps_value = 1.0 / elapsed
    _fps_last_time = now
    cv2.putText(frame, f"FPS: {int(_fps_value)}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2)

    _run_person  = (frame_count % YOLO_FRAME_MOD    == 0)
    _run_gesture = (frame_count % GESTURE_FRAME_MOD == 0)
    _run_phone   = (frame_count % 5                 == 0)
    _run_obj     = (frame_count % 6                 == 0)

    _f_person  = _yolo_pool.submit(
        person_model.track, frame,
        **dict(persist=True, tracker="bytetrack.yaml",
               imgsz=640, conf=0.35, iou=0.6,
               max_det=50, agnostic_nms=True, verbose=False)
    ) if _run_person else None

    _f_gesture = _yolo_pool.submit(
        gesture_model, frame,
        **dict(verbose=False, imgsz=320, max_det=10)
    ) if _run_gesture else None

    _f_phone   = _yolo_pool.submit(
        coco_model, frame,
        **dict(verbose=False, imgsz=416, max_det=10, classes=[67])
    ) if _run_phone else None

    _f_obj     = _yolo_pool.submit(
        object_model, frame,
        **dict(verbose=False, imgsz=320, max_det=10)
    ) if _run_obj else None

    _f_idcard  = _yolo_pool.submit(
        idcard_model, frame,
        **dict(verbose=False, imgsz=320, max_det=5)
    ) if _run_obj else None

    _f_mesh  = _mp_pool.submit(face_mesh.process,     img_rgb)
    _f_pose  = _mp_pool.submit(pose_detector.process, img_rgb)
    _f_hands = _mp_pool.submit(hand_detector.process, img_rgb)

    if _f_person is not None:
        try:
            _pr = _f_person.result()
            if _pr and len(_pr) > 0:
                last_custom_results = _pr[0]
        except Exception as _e:
            log.error("YOLO-person error: %s", _e)

    if _f_gesture is not None:
        try:
            gesture_results = _f_gesture.result()[0]
        except Exception as _e:
            log.error("YOLO-gesture error: %s", _e)
            gesture_results = None

    if _f_phone is not None:
        try:
            last_coco_results = _f_phone.result()[0]
        except Exception as _e:
            log.error("YOLO-phone error: %s", _e)

    _parallel_obj_res    = None
    _parallel_idcard_res = None
    if _f_obj is not None:
        try:
            _parallel_obj_res    = _f_obj.result()[0]
        except Exception as _e:
            log.error("YOLO-object error: %s", _e)
        try:
            _parallel_idcard_res = _f_idcard.result()[0]
        except Exception as _e:
            log.error("YOLO-idcard error: %s", _e)

    try:
        mesh_res  = _f_mesh.result()
    except Exception as _e:
        log.error("MP-mesh error: %s", _e)
        mesh_res = None

    try:
        pose_res  = _f_pose.result()
    except Exception as _e:
        log.error("MP-pose error: %s", _e)
        pose_res = None

    try:
        hand_res  = _f_hands.result()
    except Exception as _e:
        log.error("MP-hands error: %s", _e)
        hand_res = None

    if last_custom_results is not None and last_custom_results.boxes is not None:
        _invigilator_tids = set()
        all_bbox_heights = [
            int(box.xyxy[0][3]) - int(box.xyxy[0][1])
            for box in last_custom_results.boxes
            if float(box.conf[0]) >= CUSTOM_CONF_THRESHOLD
               and int(box.cls[0]) == CUSTOM_CLASS["PERSON"]
               and box.id is not None
        ]
        if all_bbox_heights:
            if _median_student_bbox_h == 0.0:
                _median_student_bbox_h = float(np.median(all_bbox_heights))
            else:
                _median_student_bbox_h = (
                    0.98 * _median_student_bbox_h
                    + 0.02 * float(np.median(all_bbox_heights))
                )

        for box in last_custom_results.boxes:
            cls, conf = int(box.cls[0]), float(box.conf[0])
            if conf < CUSTOM_CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = center(x1, y1, x2, y2)
            if cls == CUSTOM_CLASS["PERSON"] and box.id is not None:
                tid = int(box.id[0])
                bbox_h = y2 - y1
                if (_median_student_bbox_h > 0
                        and bbox_h > _median_student_bbox_h * INVIGILATOR_HEIGHT_RATIO):
                    _invigilator_tids.add(tid)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 0), 1)
                    cv2.putText(frame, "INVIG", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 0), 1)
                    continue

                old_pos = current_positions.get(tid)
                if old_pos is None:
                    if tid in current_positions:
                        px, py = current_positions[tid]
                        cx = int(px * 0.7 + cx * 0.3)
                        cy = int(py * 0.7 + cy * 0.3)
                    current_positions[tid] = (cx, cy)
                elif dist(old_pos, (cx, cy)) < 150:
                    current_positions[tid] = (
                        int(old_pos[0] * 0.6 + cx * 0.4),
                        int(old_pos[1] * 0.6 + cy * 0.4),
                    )
                current_ids.add(tid)
                sid = student_id_map.get(tid, None)

                if sid is None and locked:
                    pos = current_positions.get(tid)
                    if pos is not None:
                        claimed_sids = {student_id_map.get(t) for t in current_ids if t != tid}
                        REACQUIRE_DIST = 70
                        for s_sid, s_pos in seat_positions.items():
                            if s_sid in claimed_sids or s_sid == "S???":
                                continue
                            if dist(pos, s_pos) > REACQUIRE_DIST:
                                continue
                            if seat_vacancy_counters.get(s_sid, 0) == 0:
                                continue
                            last_ra = last_alert_time.get(f"_reacquire_{s_sid}", -999)
                            if frame_count - last_ra < 45:
                                continue
                            student_id_map[tid] = s_sid
                            last_alert_time[f"_reacquire_{s_sid}"] = frame_count
                            sid = s_sid
                            log.info("ReAcquire: New tid=%d → %s (dist=%.0fpx)",
                                     tid, s_sid, dist(pos, s_pos))
                            break

                sid_label = sid if sid else "Scanning..."
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
                cv2.putText(frame, sid_label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    if len(current_ids) == 0:
        last_coco_results = None

    for tid in current_ids:
        sid = student_id_map.get(tid)
        if sid:
            student_state.update(
                sid,
                face_present=True,
                position=current_positions.get(tid),
                gaze=gaze_history.get(sid)
            )

    if gesture_results is not None and gesture_results.boxes is not None:
        face_centers_for_gesture = get_all_face_centers(last_face_landmarks, iw, ih)

        for box in gesture_results.boxes:
            cls, conf = int(box.cls[0]), float(box.conf[0])
            if conf < GESTURE_CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            gesture_cx, gesture_cy = center(x1, y1, x2, y2)

            if point_is_on_face((gesture_cx, gesture_cy), face_centers_for_gesture,
                                 radius=GESTURE_FACE_OVERLAP_RADIUS):
                continue

            tid = get_student_at(gesture_cx, gesture_cy, current_positions)
            if tid is None:
                continue
            sid = student_id_map.get(tid)
            if sid is None:
                continue
            gesture_name = gesture_model.names[cls].lower()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(frame, f"{gesture_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
            sig_key = f"{sid}_{gesture_name}"
            if gesture_name in ["peace", "call_me", "thumbs_up", "thumbs_down", "point_up"]:
                signaling_counters[sig_key] = signaling_counters.get(sig_key, 0) + 1
                if signaling_counters[sig_key] >= SIGNAL_CONFIRM_FRAMES:
                    if can_send(f"gesture_{sig_key}", cooldown=2.0):
                        ev = event_manager.emit(sid, "SUSPICIOUS_GESTURE", conf)
                        student_state.add_event(sid, ev)
                        logger.log_event(
                            student_id=sid,
                            event="SUSPICIOUS_GESTURE",
                            confidence=conf,
                            frame=frame_count
                        )
                        send_event_async("SUSPICIOUS_GESTURE", sid)
                        score = score_engine.add_event(sid, "SUSPICIOUS_GESTURE")
                        if score > 80:
                            log.warning("CHEATING HIGH RISK: %s | score=%s", sid, score)
                    current_alert = f"SUSPICIOUS GESTURE: {sid}"
            else:
                signaling_counters[sig_key] = 0

    if last_coco_results is not None and last_coco_results.boxes is not None:
        for box in last_coco_results.boxes:
            cls, conf = int(box.cls[0]), float(box.conf[0])
            if cls != PHONE_CLASS_ID or conf < PHONE_CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            w_box, h_box = x2 - x1, y2 - y1
            area = w_box * h_box
            ar   = h_box / float(w_box + 1e-6)
            is_portrait  = ar >= 0.9
            is_landscape = ar <= 0.85 and h_box >= 12
            if not (PHONE_MIN_AREA < area < PHONE_MAX_AREA):
                continue
            if not (is_portrait or is_landscape) or w_box < 12:
                continue
            cx, cy = center(x1, y1, x2, y2)
            skip_phone = False
            for fl in (last_face_landmarks or []):
                fc = get_face_center(fl, iw, ih)
                if dist((cx, cy), fc) < 25:
                    skip_phone = True
                    break
            if skip_phone:
                continue

            phones.append((cx, cy))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"PHONE {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    object_results = _parallel_obj_res
    if _parallel_idcard_res is not None and _parallel_idcard_res.boxes is not None:
        for box in _parallel_idcard_res.boxes:
            cls, conf = int(box.cls[0]), float(box.conf[0])
            if conf < IDCARD_CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            w_id, h_id = x2 - x1, y2 - y1
            if h_id / float(w_id + 1e-6) < 1.0:
                continue
            tid = get_student_at(*center(x1, y1, x2, y2), current_positions)
            if tid is None:
                continue
            sid = student_id_map.get(tid)
            if sid is None:
                continue
            id_detected_this_frame.add(sid)
            id_visible_counters[sid] = 0
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"ID CARD {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    if not locked:
        stable_counter = stable_counter + 1 if len(current_ids) > 0 else 0

        for tid, pos in current_positions.items():
            if tid not in _calib_accumulated_positions:
                _calib_accumulated_positions[tid] = pos
            else:
                old = _calib_accumulated_positions[tid]
                _calib_accumulated_positions[tid] = (
                    int(old[0] * 0.7 + pos[0] * 0.3),
                    int(old[1] * 0.7 + pos[1] * 0.3),
                )

        calib_samples = learner.on_calibration_frame(
            img_rgb, current_positions, face_mesh, pose_detector, iw, ih
        )
        progress = int((stable_counter / STABILITY_FRAMES) * (iw - 40))
        cv2.rectangle(frame, (20, ih - 30), (iw - 20, ih - 10), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, ih - 30), (20 + progress, ih - 10), (0, 255, 255), -1)
        cv2.putText(
            frame,
            f"CALIBRATING... ({stable_counter}/{STABILITY_FRAMES}) | "
            f"seen: {len(_calib_accumulated_positions)} students | samples: {calib_samples}",
            (20, ih - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
        )
        if stable_counter >= STABILITY_FRAMES:
            locked      = True
            _lock_frame = frame_count
            all_positions = _calib_accumulated_positions
            matched_map   = learner.match_seats(all_positions)
            if matched_map:
                student_id_map.update(matched_map)
                log.info("Seat-based matching: %d matched.", len(matched_map))
            else:
                student_id_map.update(assign_student_ids(all_positions))
                log.info("Fresh IDs assigned to %d students.", len(all_positions))
            learner.on_lock(student_id_map)

            seat_anchor_positions = dict(_calib_accumulated_positions)
            log.info("Seat anchors: %d positions locked", len(seat_anchor_positions))
            init_seat_zones(seat_anchor_positions)
            learner.set_seat_positions(seat_positions)
            log.info("LOCKED — %d student(s): %s", len(student_id_map), list(student_id_map.values()))
            log.info("Seat zones: %d", len(seat_positions))
        return frame

    _settled = (frame_count - _lock_frame) > _SETTLE_FRAMES
    if frame_count % 30 == 0:
        prune_state(current_ids)
    draw_seat_zones(frame)

    zone_alert = check_seat_zones(current_positions, frame)
    if zone_alert:
        current_alert = zone_alert

    ID_VISIBLE_THRESHOLD = 99999
    for tid in current_ids:
        sid = student_id_map.get(tid)
        if sid is None:
            continue
        id_visible_counters.setdefault(sid, 0)
        if sid in id_detected_this_frame:
            id_visible_counters[sid] = 0
        else:
            id_visible_counters[sid] += 1
        if id_visible_counters[sid] >= ID_VISIBLE_THRESHOLD:
            if can_send(f"id_missing_{sid}", cooldown=30):
                ev = event_manager.emit(sid, "ID_NOT_VISIBLE", 0.9)
                student_state.add_event(sid, ev)
                send_event_async("ID_NOT_VISIBLE", sid)
                score = score_engine.add_event(sid, "ID_NOT_VISIBLE")
                if score > 80:
                    log.warning("CHEATING HIGH RISK: %s | score=%s", sid, score)
            current_alert = f"ID NOT VISIBLE: {sid}"

    if mesh_res is None and (frame_count - last_face_landmarks_frame) > FACE_ABSENT_FRAMES_THRESHOLD:
        try:
            mesh_res = face_mesh.process(img_rgb)
        except Exception:
            pass

    if pose_res and pose_res.pose_landmarks:
        lms = pose_res.pose_landmarks.landmark
        tid = get_student_at(int(lms[0].x * iw), int(lms[0].y * ih), current_positions)
        sid = student_id_map.get(tid) if tid is not None else None
        if sid is not None:
            raw_lean  = abs(lms[11].x - lms[12].x)
            prev      = _lean_smooth.get(sid, raw_lean)
            _lean_smooth[sid] = 0.5 * raw_lean + 0.5 * prev
            smooth_lean       = _lean_smooth[sid]
            leaning_detected  = learner.check_lean(sid, smooth_lean)
            if _settled and learner._stable_trigger(sid, "BODY_LEANING", leaning_detected):
                current_alert = f"LEANING: {sid}"
                if can_send(f"lean_{sid}", cooldown=20.0):
                    save_evidence(frame, sid, "BODY_LEANING")
                    learner.register_event(sid, "BODY_LEANING")
                    ev = event_manager.emit(sid, "BODY_LEANING", 0.9)
                    student_state.add_event(sid, ev)
                    logger.log_event(
                        student_id=sid,
                        event="BODY_LEANING",
                        confidence=0.9,
                        frame=frame_count
                    )
                    send_event_async("BODY_LEANING", sid)
                    score = score_engine.add_event(sid, "BODY_LEANING")
                    if score > 80:
                        log.warning("CHEATING HIGH RISK: %s | score=%s", sid, score)

    if mesh_res and mesh_res.multi_face_landmarks:
        last_face_landmarks       = mesh_res.multi_face_landmarks
        last_face_landmarks_frame = frame_count
        current_talking           = set()

        for f_lms in mesh_res.multi_face_landmarks:
            pts = f_lms.landmark
            fx, fy = int(pts[1].x * iw), int(pts[1].y * ih)
            tid = get_student_at(fx, fy, current_positions)
            if tid is None:
                continue
            prev = face_based_positions.get(tid, (fx, fy))
            face_based_positions[tid] = (
                int(0.2 * fx + 0.8 * prev[0]),
                int(0.2 * fy + 0.8 * prev[1]),
            )

            sid = student_id_map.get(tid, "Unknown")

            mar = mouth_aspect_ratio(pts)
            if hand_face_counters.get(sid, 0) > 0 or hand_mouth_counters.get(sid, 0) > 0:
                mar = 0.0
            history = mar_history.get(sid, [])
            history.append(mar)
            if len(history) > MAR_WINDOW_SIZE:
                history.pop(0)
            mar_history[sid] = history
            smoothed  = sum(history) / len(history)
            variation = max(history) - min(history) if len(history) > 2 else 0
            mouth_open = is_mouth_open_vertically(pts)

            look_down_raw = get_look_down_score(pts)
            gaze_ratio, looking_left, looking_right = get_gaze_direction(pts)
            head_ratio = pts[1].x
            head_ratio_history[sid] = head_ratio

            prev_gaze    = gaze_history.get(sid, gaze_ratio)
            gaze_ratio   = 0.5 * prev_gaze + 0.5 * gaze_ratio
            gaze_history[sid] = gaze_ratio

            result = learner.on_detection_frame(
                sid,
                look_down_raw,
                gaze_ratio,
                smoothed,
                variation,
                mouth_open,
            )

            result.looking_left  = result.looking_left  or looking_left
            result.looking_right = result.looking_right or looking_right

            if _settled and learner._stable_trigger(sid, "LOOKING_DOWN", result.looking_down):
                current_alert = f"LOOKING DOWN: {sid}"
                if can_send(f"down_{sid}"):
                    save_evidence(frame, sid, "LOOKING_DOWN")
                    learner.register_event(sid, "LOOKING_DOWN")
                    ev = event_manager.emit(sid, "LOOKING_DOWN", 0.9)
                    student_state.add_event(sid, ev)
                    logger.log_event(
                        student_id=sid,
                        event="LOOKING_DOWN",
                        confidence=0.9,
                        frame=frame_count
                    )
                    send_event_async("LOOKING_DOWN", sid)
                    score = score_engine.add_event(sid, "LOOKING_DOWN")
                    if score > 80:
                        log.warning("CHEATING HIGH RISK: %s | score=%s", sid, score)
                if tid in current_positions:
                    for p_pos in phones:
                        if dist(current_positions[tid], p_pos) < 80:
                            if can_send(f"phonegaze_{sid}"):
                                save_evidence(frame, sid, "LOOKING_AT_PHONE")
                                learner.register_event(sid, "LOOKING_AT_PHONE")
                                ev = event_manager.emit(sid, "LOOKING_AT_PHONE", 0.9)
                                student_state.add_event(sid, ev)
                                logger.log_event(
                                    student_id=sid,
                                    event="LOOKING_AT_PHONE",
                                    confidence=0.9,
                                    frame=frame_count
                                )
                                send_event_async("LOOKING_AT_PHONE", sid)
                                score = score_engine.add_event(sid, "LOOKING_AT_PHONE")
                                if score > 80:
                                    log.warning("CHEATING HIGH RISK: %s | score=%s", sid, score)

            if _settled and learner._stable_trigger(sid, "LOOKING_LEFT", result.looking_left):
                current_alert = f"LOOKING LEFT: {sid}"
                if can_send(f"left_{sid}"):
                    save_evidence(frame, sid, "LOOKING_LEFT")
                    learner.register_event(sid, "LOOKING_LEFT")
                    ev = event_manager.emit(sid, "LOOKING_LEFT", 0.9)
                    student_state.add_event(sid, ev)
                    logger.log_event(
                        student_id=sid,
                        event="LOOKING_LEFT",
                        confidence=0.9,
                        frame=frame_count
                    )
                    send_event_async("LOOKING_LEFT", sid)
                    score = score_engine.add_event(sid, "LOOKING_LEFT")
                    if score > 80:
                        log.warning("CHEATING HIGH RISK: %s | score=%s", sid, score)

            elif _settled and learner._stable_trigger(sid, "LOOKING_RIGHT", result.looking_right):
                current_alert = f"LOOKING RIGHT: {sid}"
                if can_send(f"right_{sid}"):
                    save_evidence(frame, sid, "LOOKING_RIGHT")
                    learner.register_event(sid, "LOOKING_RIGHT")
                    ev = event_manager.emit(sid, "LOOKING_RIGHT", 0.9)
                    student_state.add_event(sid, ev)
                    logger.log_event(
                        student_id=sid,
                        event="LOOKING_RIGHT",
                        confidence=0.9,
                        frame=frame_count
                    )
                    send_event_async("LOOKING_RIGHT", sid)
                    score = score_engine.add_event(sid, "LOOKING_RIGHT")
                    if score > 80:
                        log.warning("CHEATING HIGH RISK: %s | score=%s", sid, score)

            if result.talking:
                mouth_counters[sid] = mouth_counters.get(sid, 0) + TALK_COUNTER_INCREMENT
            else:
                mouth_counters[sid] = max(0.0, mouth_counters.get(sid, 0) - TALK_COUNTER_DECAY)

            if mouth_counters.get(sid, 0) >= MAR_TALK_FRAMES:
                current_talking.add(sid)
                if _settled and can_send(f"talking_{sid}", cooldown=8.0):
                    ev = event_manager.emit(sid, "TALKING_DETECTED", 0.85)
                    student_state.add_event(sid, ev)
                    logger.log_event(
                        student_id=sid,
                        event="TALKING_DETECTED",
                        confidence=0.85,
                        frame=frame_count
                    )
                    send_event_async("TALKING_DETECTED", sid)
                    score_engine.add_event(sid, "TALKING_DETECTED")
                    current_alert = f"TALKING: {sid}"

            if tid in current_positions:
                cx2, cy2 = current_positions[tid]
                status = "TALKING" if sid in current_talking else f"MAR:{smoothed:.3f}"
                color  = (0, 255, 0) if sid in current_talking else (0, 255, 255)
                cv2.putText(frame, status, (cx2 + 10, cy2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                calib_tag = "V" if result.is_calibrated else "~"
                cv2.putText(
                    frame,
                    f"{calib_tag} dw:{result.dev_look_down:+.3f} yw:{result.dev_yaw:+.3f}",
                    (cx2 + 10, cy2 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 220, 255), 1,
                )

        last_talking_students = current_talking

    face_landmarks_list = last_face_landmarks if last_face_landmarks else []

    for tid in current_ids:
        sid = student_id_map.get(tid, "Unknown")
        pos = current_positions[tid]

        for p_pos in phones:
            if PHONE_PROXIMITY_MIN_DIST < dist(pos, p_pos) < PHONE_PROXIMITY_DIST:
                phone_counters[sid] = phone_counters.get(sid, 0) + 1
                if phone_counters[sid] >= 2:
                    if can_send(f"phone_{sid}"):
                        save_evidence(frame, sid, "PHONE_DETECTED")
                        learner.register_event(sid, "PHONE_DETECTED")
                        ev = event_manager.emit(sid, "PHONE_DETECTED", 0.9)
                        student_state.add_event(sid, ev)
                        logger.log_event(
                            student_id=sid,
                            event="PHONE_DETECTED",
                            confidence=0.9,
                            frame=frame_count
                        )
                        send_event_async("PHONE_DETECTED", sid)
                        score = score_engine.add_event(sid, "PHONE_DETECTED")
                        if score > 80:
                            log.warning("CHEATING HIGH RISK: %s | score=%s", sid, score)
            else:
                phone_counters[sid] = 0

        for other_tid in current_ids:
            if tid >= other_tid:
                continue
            sid_other = student_id_map.get(other_tid, "Unknown")
            d = dist(pos, current_positions[other_tid])
            _either_talking = (
                mouth_counters.get(sid, 0)       >= MAR_TALK_FRAMES or
                mouth_counters.get(sid_other, 0) >= MAR_TALK_FRAMES
            )
            if _either_talking and d < WHISPER_PROXIMITY_THRESHOLD:
                cv2.line(frame, pos, current_positions[other_tid], (0, 0, 255), 2)
                current_alert = f"WHISPER: {sid} & {sid_other}"
                pair_key = f"whisp_{min(sid, sid_other)}_{max(sid, sid_other)}"
                if can_send(pair_key):
                    ev = event_manager.emit(f"{sid}-{sid_other}", "WHISPERING_DETECTED", 0.9)
                    student_state.add_event(sid, ev)
                    logger.log_event(
                        student_id=f"{sid}-{sid_other}",
                        event="WHISPERING_DETECTED",
                        confidence=0.9,
                        frame=frame_count
                    )
                    send_event_async("WHISPERING_DETECTED", f"{sid}-{sid_other}")
                    score = score_engine.add_event(sid, "WHISPERING_DETECTED")
                    if score > 80:
                        log.warning("CHEATING HIGH RISK: %s | score=%s", sid, score)
                    learner.escalate_if_coordinated(sid, sid_other)

    if object_results is not None and object_results.boxes is not None:
        for box in object_results.boxes:
            cls, conf = int(box.cls[0]), float(box.conf[0])
            label     = object_model.names[cls].lower()

            min_conf = OBJECT_CLASS_CONF.get(label, 0.60)
            if conf < min_conf:
                continue
            if label == "person":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            obj_cx, obj_cy = center(x1, y1, x2, y2)

            sid = get_seat_at(obj_cx, obj_cy) if seat_positions else None
            if sid is None:
                tid = get_student_at(obj_cx, obj_cy, current_positions)
                sid = student_id_map.get(tid) if tid else None
            if sid is None:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            if label == "calculator":
                if sid not in calc_seen_window:
                    calc_seen_window[sid] = _deque(maxlen=CALC_PERSIST_WINDOW)
                calc_seen_window[sid].append(1)
                presence_rate = sum(calc_seen_window[sid]) / len(calc_seen_window[sid])

                if can_send(f"calc_{sid}", cooldown=CALC_PERSIST_COOLDOWN):
                    save_evidence(frame, sid, "CALCULATOR_DETECTED")
                    learner.register_event(sid, "CALCULATOR_DETECTED")
                    ev = event_manager.emit(sid, "CALCULATOR_DETECTED", conf)
                    student_state.add_event(sid, ev)
                    logger.log_event(
                        student_id=sid,
                        event="CALCULATOR_DETECTED",
                        confidence=conf,
                        frame=frame_count
                    )
                    send_event_async("CALCULATOR_DETECTED", sid)
                    score = score_engine.add_event(sid, "CALCULATOR_DETECTED")
                    if score > 80:
                        log.warning("CHEATING HIGH RISK: %s | score=%s", sid, score)

                if (len(calc_seen_window[sid]) >= CALC_PERSIST_WINDOW
                        and presence_rate >= CALC_PERSIST_RATE):
                    if can_send(f"calc_persist_{sid}", cooldown=CALC_PERSIST_COOLDOWN):
                        save_evidence(frame, sid, "CALCULATOR_PERSISTENT")
                        ev = event_manager.emit(sid, "CALCULATOR_PERSISTENT", round(presence_rate, 2))
                        student_state.add_event(sid, ev)
                        logger.log_event(
                            student_id=sid,
                            event="CALCULATOR_PERSISTENT",
                            confidence=round(presence_rate, 2),
                            frame=frame_count
                        )
                        send_event_async("CALCULATOR_PERSISTENT", sid)
                        score = score_engine.add_event(sid, "CALCULATOR_DETECTED")
                        if score > 80:
                            log.warning("CHEATING HIGH RISK: %s | score=%s", sid, score)

                current_alert = f"CALCULATOR DETECTED: {sid}"

    if hand_res:
        hand_alert = process_hands(
            hand_res, face_landmarks_list, current_positions, iw, ih, frame, last_talking_students
        )
        if hand_alert and (current_alert == "All Clear" or "PASSING" in hand_alert):
            current_alert = hand_alert

    _all_calibrated = all(
        learner.is_calibrated(student_id_map[tid])
        for tid in current_ids if tid in student_id_map
    ) if current_ids else False
    # Only top-up calibration if we have uncalibrated students AND
    # we haven't already locked from an external snapshot phase
    if not _all_calibrated and frame_count % 2 == 0 and frame_count < 300:
        learner.on_calibration_frame(
            img_rgb, current_positions, face_mesh, pose_detector, iw, ih,
            mesh_res=mesh_res, pose_res=pose_res
        )

    head_direction = "forward"
    if "LOOKING LEFT" in current_alert:
        head_direction = "left"
    elif "LOOKING RIGHT" in current_alert:
        head_direction = "right"
    elif "LOOKING DOWN" in current_alert:
        head_direction = "down"

    hands_visible   = hand_res is not None and hand_res.multi_hand_landmarks is not None
    student_hands_visible = hands_visible
    if hands_visible and _invigilator_tids and hand_res.multi_hand_landmarks:
        student_hands_visible = False
        student_positions_set = {
            current_positions[t] for t in current_ids
            if t not in _invigilator_tids and t in current_positions
        }
        for hand_lms in hand_res.multi_hand_landmarks:
            wrist = hand_lms.landmark[0]
            wx, wy = int(wrist.x * iw), int(wrist.y * ih)
            if any(dist((wx, wy), sp) < 150 for sp in student_positions_set):
                student_hands_visible = True
                break

    current_gesture = "none"
    if gesture_results is not None and gesture_results.boxes is not None:
        if len(gesture_results.boxes) > 0:
            current_gesture = "gesture"

    detected_objects = []
    if len(phones) > 0:
        detected_objects.append("cell phone")

    person_count = len(current_ids)

    calc_detected_this_frame = {
        student_id_map.get(tid)
        for tid in current_ids
        if student_id_map.get(tid) in calc_seen_window
        and calc_seen_window[student_id_map.get(tid)]
        and calc_seen_window[student_id_map.get(tid)][-1] == 1
    }
    for sid_w, win in calc_seen_window.items():
        if sid_w not in calc_detected_this_frame:
            win.append(0)

    features = collector.collect(
        head_pose=head_direction,
        hands_detected=hands_visible,
        gesture=current_gesture,
        objects_detected=detected_objects,
        person_count=person_count
    )
    features["student_hands_detected"] = student_hands_visible

    intel_alerts = intel.update(
        frame_count=frame_count,
        locked=locked,
        current_positions=current_positions,
        student_id_map=student_id_map,
        seat_positions=seat_positions,
        invigilator_tids=_invigilator_tids,
        face_landmarks=last_face_landmarks,
        gaze_ratios=gaze_history,
        head_ratios=head_ratio_history,
        phones=phones,
        iw=iw, ih=ih,
        can_send_fn=can_send,
        log_event_fn=logger.log_event,
        send_async_fn=send_event_async,
    )
    for event in intel_alerts:
        # intel.update() may return dicts or plain strings — handle both
        if isinstance(event, dict):
            sid   = event.get("student_id")
            etype = event.get("event_type")
            conf  = event.get("confidence", 1.0)
        elif isinstance(event, str):
            sid   = None
            etype = event
            conf  = 1.0
        else:
            continue
        if not etype:
            continue
        ev = event_manager.emit(sid, etype, conf)
        if sid:
            student_state.add_event(sid, ev)

    classification_results = {}

    for sid in student_id_map.values():
        events = event_manager.get_recent_events(sid, 60)
        result = classifier.classify(sid, events)

        classification_results[sid] = result

        if sid in current_positions:
            x, y = current_positions[
                next(t for t in student_id_map if student_id_map[t] == sid)
            ]

            color = (0, 255, 0)
            if result["label"] == "SUSPICIOUS":
                color = (0, 255, 255)
            elif result["label"] == "CHEATING":
                color = (0, 0, 255)

            cv2.putText(
                frame,
                f'{result["label"]} ({result["score"]})',
                (x, y + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    metrics = metrics_engine.update(features)

    if metrics:
        metrics["multiple_people_rate"] = 0.0
        anomalies = anomaly_detector.detect(metrics)
        if frame_count % 30 == 0 and metrics.get("phone_presence_rate", 0) > 0:
            log.debug("Behavior Metrics: %s", {k: f"{v:.3f}" for k, v in metrics.items()})
        if anomalies:
            for a in anomalies:
                metric   = a["metric"]
                severity = a["severity"]
                value    = a["value"]
                if frame_count % 30 == 0:
                    log.warning("ANOMALY → %s | value=%.3f | severity=%s", metric, value, severity)
                room_key = f"anomaly_room_{metric}"
                if can_send(room_key, cooldown=15):
                    target_sid = None
                    for tid in current_ids:
                        sid = student_id_map.get(tid)
                        if sid:
                            target_sid = sid
                            break
                    if target_sid:
                        logger.log_event(
                            student_id=f"ROOM({metric})",
                            event="BEHAVIOR_ANOMALY",
                            confidence=value,
                            frame=frame_count
                        )
                        send_event_async("BEHAVIOR_ANOMALY", f"room_{metric}")

    if current_alert == "All Clear":
        draw_alert_banner(frame, "All Clear", color=(0, 130, 0))
    else:
        draw_alert_banner(frame, f"{current_alert}", color=(0, 0, 200))

    cv2.rectangle(frame, (0, ih - 25), (iw, ih), (30, 30, 30), -1)
    cv2.putText(
        frame,
        f"Students: {len(current_ids)}  |  Seats: {len(seat_positions)}  |  "
        f"Phones: {len(phones)}",
        (8, ih - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1,
    )
    learner.draw_debug(frame, iw)
    learner.decay_scores()

    if frame_count % 60 == 0:
        score_engine.decay_scores()

    return frame


def start_frame_reader(cap):
    def reader():
        while True:
            ret, frame = cap.read()
            if not ret:
                frame_queue.put(None)
                break
            if not frame_queue.full():
                frame_queue.put(frame)
    threading.Thread(target=reader, daemon=True).start()


def process_video(cap, out):
    start_frame_reader(cap)
    while True:
        try:
            frame = frame_queue.get(timeout=1.0)
        except Empty:
            continue
        if frame is None:
            break
        processed = run_ai_on_frame(frame)
        out.write(processed)

def generate_final_report():
    path = report_generator.generate(
        event_manager=event_manager,
        student_state=student_state,
        classifier=classifier,
        evidence_manager=evidence_manager
    )
    log.info("Final Report: %s", path)

def get_all_events():
    return event_manager.get_all_events()

def get_all_states():
    return student_state.get_all()

def get_all_evidence():
    return evidence_manager.get_all_evidence()


def shutdown():
    log.info("Shutdown — flushing alert logger...")
    try:
        logger.shutdown()
    except Exception as e:
        log.error("AlertLogger shutdown error: %s", e)

    log.info("Shutdown — flushing evidence writer...")
    try:
        _evidence_queue.join()
        _evidence_queue.put(None)
        _evidence_writer_thread.join(timeout=5.0)
    except Exception as e:
        log.error("Evidence writer shutdown error: %s", e)

    face_mesh.close()
    pose_detector.close()
    hand_detector.close()
    learner.shutdown()

    log.info("Shutdown — closing executor pools...")
    _yolo_pool.shutdown(wait=False)
    _mp_pool.shutdown(wait=False)
    log.info("Shutdown complete.")