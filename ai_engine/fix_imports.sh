#!/bin/bash
# fix_imports.sh
# Run this from your ai_engine/ folder:
#   bash fix_imports.sh

BASE="/home/tx0978/Documents/AkshatSrivastavaTX0978/LIVE_CLASS_PROCTORING_USING_AI/ai_engine"
CORE="$BASE/core"

echo "======================================================"
echo "  Fixing import structure..."
echo "======================================================"

# ── Step 1: Add missing tracking/ folder + student_state.py ───────────────
mkdir -p "$BASE/tracking"

cat > "$BASE/tracking/__init__.py" << 'EOF'
EOF

cat > "$BASE/tracking/student_state.py" << 'EOF'
# tracking/student_state.py
import time
from collections import defaultdict

class StudentState:
    def __init__(self):
        self._states = {}

    def update(self, sid, face_present=True, position=None, gaze=None):
        if sid not in self._states:
            self._states[sid] = {
                "face_present": face_present,
                "position": position,
                "gaze": gaze,
                "last_seen": time.time(),
                "events": []
            }
        else:
            s = self._states[sid]
            s["face_present"] = face_present
            if position is not None:
                s["position"] = position
            if gaze is not None:
                s["gaze"] = gaze
            s["last_seen"] = time.time()

    def add_event(self, sid, event):
        if sid not in self._states:
            self._states[sid] = {"events": [], "last_seen": time.time()}
        self._states[sid].setdefault("events", []).append(event)

    def get(self, sid):
        return self._states.get(sid, {})

    def get_all(self):
        return dict(self._states)

    def reset(self):
        self._states.clear()
EOF

echo "  [OK] tracking/student_state.py created"

# ── Step 2: Add missing model_loader.py at root ai_engine level ───────────
cat > "$BASE/model_loader.py" << 'EOF'
# model_loader.py  (root level — wraps core/model_loader if needed)
import os
from ultralytics import YOLO

MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "models", "production"
)

def load_models():
    print("[AI] Loading models...")
    person_model  = YOLO(os.path.join(MODEL_DIR, "person_model.pt"))
    gesture_model = YOLO(os.path.join(MODEL_DIR, "gesture_model.pt"))
    object_model  = YOLO(os.path.join(MODEL_DIR, "object_model.pt"))
    idcard_model  = YOLO(os.path.join(MODEL_DIR, "id_model.pt"))
    print("[AI] All models loaded.")
    return person_model, gesture_model, object_model, idcard_model
EOF

echo "  [OK] model_loader.py created at root"

# ── Step 3: Fix core/ai_engine.py imports ─────────────────────────────────
# All the broken flat imports need to become subpackage-relative.
# We do a series of sed replacements on the file.

AI="$CORE/ai_engine.py"

# 3a — flat → subpackage imports
sed -i \
  -e 's/^from adaptive_learning import AdaptiveLearner$/from core.adaptive_learning import AdaptiveLearner/' \
  -e 's/^from alert_logger import AlertLogger$/from output.alert_logger import AlertLogger/' \
  -e 's/^from feature_collector import FeatureCollector$/from output.feature_collector import FeatureCollector/' \
  -e 's/^from metrics_engine import MetricsEngine$/from detection.metrics_engine import MetricsEngine/' \
  -e 's/^from anomaly_detector import AnomalyDetector$/from detection.anomaly_detector import AnomalyDetector/' \
  -e 's/^from model_loader import load_models$/from model_loader import load_models/' \
  -e 's/^from cheating_score import CheatingScore$/from classification.cheating_score import CheatingScore/' \
  -e 's/^from core\.event_manager import EventManager$/from core.event_manager import EventManager/' \
  -e 's/^from tracking\.student_state import StudentState$/from tracking.student_state import StudentState/' \
  -e 's/^from output\.evidence_manager import EvidenceManager$/from output.evidence_manager import EvidenceManager/' \
  -e 's/^from detection\.intelligence_layer import IntelligenceLayer$/from detection.intelligence_layer import IntelligenceLayer/' \
  -e 's/^from classification\.cheating_classifier import CheatingClassifier$/from classification.cheating_classifier import CheatingClassifier/' \
  -e 's/^from output\.report_generator import ReportGenerator$/from output.report_generator import ReportGenerator/' \
  "$AI"

echo "  [OK] core/ai_engine.py imports fixed"

# 3b — Fix the broken config.get() lines (Patch 1)
sed -i \
  -e 's/^PHONE_CONF_THRESHOLD = config\.get.*$/PHONE_CONF_THRESHOLD = 0.45/' \
  -e 's/^GAZE_LEFT_THRESHOLD = config\.get.*$/GAZE_LEFT_THRESHOLD  = 0.42/' \
  -e 's/^AI_FRAME_SIZE = tuple(config\.get.*$/AI_FRAME_SIZE        = (960, 540)/' \
  "$AI"

echo "  [OK] config.get() crash lines fixed"

# ── Step 4: Add __init__.py to all subpackage folders ─────────────────────
for dir in classification detection output tracking; do
    touch "$BASE/$dir/__init__.py"
    echo "  [OK] $dir/__init__.py ensured"
done

# ── Step 5: Create root-level ai_engine.py shim ───────────────────────────
# process_video.py and test_engine.py do:  from ai_engine import run_ai_on_frame
# But ai_engine.py is in core/. This shim re-exports everything.
cat > "$BASE/ai_engine.py" << 'EOF'
"""
ai_engine.py  (root-level shim)
Re-exports everything from core.ai_engine so that:
  - process_video.py  can do:  import ai_engine as ae
  - test_engine.py    can do:  from ai_engine import run_ai_on_frame
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))   # ensure ai_engine/ is on path

from core.ai_engine import *
from core.ai_engine import (
    run_ai_on_frame,
    reset_session,
    shutdown,
    process_video,
    generate_final_report,
    get_all_events,
    get_all_states,
    get_all_evidence,
    send_event_async,
    # module-level state that process_video.py writes to
    EXAM_ID,
    ROOM_ID,
    ALERT_URL,
    AI_FRAME_SIZE,
    SEAT_ZONE_RADIUS,
    STABILITY_FRAMES,
    CUSTOM_CONF_THRESHOLD,
    CUSTOM_CLASS,
)

# Re-export mutable module-level objects
import core.ai_engine as _core
score_engine    = _core.score_engine
learner         = _core.learner
person_model    = _core.person_model
pose_detector   = _core.pose_detector
student_id_map  = _core.student_id_map
seat_positions  = _core.seat_positions

def __getattr__(name):
    return getattr(_core, name)
EOF

echo "  [OK] Root-level ai_engine.py shim created"

# ── Step 6: Fix test_engine.py to add cap.release() ──────────────────────
cat > "$BASE/test_engine.py" << 'EOF'
import cv2
from ai_engine import run_ai_on_frame

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam. Try index 1 or 2.")
    exit(1)

print("[INFO] Press ESC to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Frame not received.")
        break

    out = run_ai_on_frame(frame)
    cv2.imshow("AI Proctor", out)

    if cv2.waitKey(1) == 27:   # ESC
        break

cap.release()
cv2.destroyAllWindows()
EOF

echo "  [OK] test_engine.py fixed"

echo ""
echo "======================================================"
echo "  All fixes applied. Now run:"
echo ""
echo "  cd $BASE"
echo ""
echo "  # Test with webcam:"
echo "  python3 test_engine.py"
echo ""
echo "  # Test with video file:"
echo "  python3 process_video.py Student_Exam_Cheating_CCTV_Video.mp4"
echo "======================================================"
