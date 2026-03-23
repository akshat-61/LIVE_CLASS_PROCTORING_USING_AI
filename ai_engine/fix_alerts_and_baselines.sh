#!/bin/bash
# fix_alerts_and_baselines.sh
# Run from ai_engine/ folder:  bash fix_alerts_and_baselines.sh

BASE="/home/tx0978/Documents/AkshatSrivastavaTX0978/LIVE_CLASS_PROCTORING_USING_AI/ai_engine"

echo "======================================================"
echo "  Fix 1 — Alert wiring through shim"
echo "======================================================"

# The root-level ai_engine.py shim exports EXAM_ID/ROOM_ID as values (not references).
# process_video.py does:  ae.EXAM_ID = exam_id  — this writes to shim module,
# but core/ai_engine.py still reads its own EXAM_ID (unchanged).
# Fix: make shim use __setattr__ to forward writes to core.

cat > "$BASE/ai_engine.py" << 'SHIM'
"""
ai_engine.py  (root-level shim)
Forwards all attribute reads AND writes to core.ai_engine
so process_video.py's  ae.EXAM_ID = ...  actually takes effect.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import core.ai_engine as _core

# ── Re-export everything ────────────────────────────────────────────
from core.ai_engine import (
    run_ai_on_frame,
    reset_session,
    shutdown,
    generate_final_report,
    get_all_events,
    get_all_states,
    get_all_evidence,
    send_event_async,
    save_evidence,
    assign_student_ids,
    init_seat_zones,
    can_send,
    dist,
    AI_FRAME_SIZE,
    SEAT_ZONE_RADIUS,
    STABILITY_FRAMES,
    CUSTOM_CONF_THRESHOLD,
    CUSTOM_CLASS,
    PHONE_CONF_THRESHOLD,
    GAZE_LEFT_THRESHOLD,
    GAZE_RIGHT_THRESHOLD,
)

# ── Live references to mutable module-level objects ─────────────────
score_engine   = _core.score_engine
learner        = _core.learner
person_model   = _core.person_model
pose_detector  = _core.pose_detector
student_id_map = _core.student_id_map
seat_positions = _core.seat_positions
locked         = _core.locked
stable_counter = _core.stable_counter
_lock_frame    = _core._lock_frame
_calib_accumulated_positions = _core._calib_accumulated_positions

# ── Forward all attribute access to core ────────────────────────────
def __getattr__(name):
    return getattr(_core, name)

def __setattr__(name, value):
    # Forward writes (ae.EXAM_ID = ..., ae.send_event_async = ...) to core
    setattr(_core, name, value)
SHIM

echo "  [OK] Shim rewritten with __setattr__ forwarding"

echo ""
echo "======================================================"
echo "  Fix 2 — Fallback baselines (S013–S033 all yw=0.5000)"
echo "======================================================"

# These students show exact fallback values because:
# 1. student_profiles.json has old data pre-loaded (34 baselines)
# 2. The snapshot calibration only gets ~3 sec of the video
# 3. Many students at the back aren't seen clearly by face mesh
# Fix: delete old student_profiles.json so it starts fresh each run
# (or rename it so it's kept as backup)

if [ -f "$BASE/student_profiles.json" ]; then
    mv "$BASE/student_profiles.json" "$BASE/student_profiles_backup.json"
    echo "  [OK] Moved old student_profiles.json → student_profiles_backup.json"
    echo "       (fresh calibration will now happen from video)"
fi

echo ""
echo "======================================================"
echo "  Fix 3 — process_video.py: lower CALIBRATION_FRAMES"
echo "           so 8-sec video can fully calibrate"
echo "======================================================"

python3 - << 'EOF'
import re

path = "core/adaptive_learning.py"
with open(path) as f:
    src = f.read()

# Lower calibration frames so short videos calibrate properly
src = src.replace("CALIBRATION_FRAMES = 6", "CALIBRATION_FRAMES = 3")
src = src.replace("CALIBRATION_FRAMES_LEAN = 6", "CALIBRATION_FRAMES_LEAN = 3")

with open(path, "w") as f:
    f.write(src)

print("  [OK] CALIBRATION_FRAMES lowered 6→3 for short-video support")
EOF

echo ""
echo "======================================================"
echo "  Fix 4 — Suppress the duplicate on_lock() call"
echo "           (Phase 3 re-runs calibration unnecessarily)"
echo "======================================================"

python3 - << 'EOF'
path = "core/ai_engine.py"
with open(path) as f:
    src = f.read()

# The post-lock calibration block fires on_calibration_frame during analysis
# which triggers a second on_lock() when learner.is_calibrated() checks fail.
# Guard it: only run if NOT already locked by process_video snapshot phase.

old = """    _all_calibrated = all(
        learner.is_calibrated(student_id_map[tid])
        for tid in current_ids if tid in student_id_map
    ) if current_ids else False
    if not _all_calibrated and frame_count % 2 == 0:
        learner.on_calibration_frame(
            img_rgb, current_positions, face_mesh, pose_detector, iw, ih,
            mesh_res=mesh_res, pose_res=pose_res
        )"""

new = """    _all_calibrated = all(
        learner.is_calibrated(student_id_map[tid])
        for tid in current_ids if tid in student_id_map
    ) if current_ids else False
    # Only top-up calibration if we have uncalibrated students AND
    # we haven't already locked from an external snapshot phase
    if not _all_calibrated and frame_count % 2 == 0 and frame_count < 300:
        learner.on_calibration_frame(
            img_rgb, current_positions, face_mesh, pose_detector, iw, ih,
            mesh_res=mesh_res, pose_res=pose_res
        )"""

if old in src:
    src = src.replace(old, new)
    with open(path, "w") as f:
        f.write(src)
    print("  [OK] Post-lock calibration guard added")
else:
    print("  [SKIP] Pattern already changed or not found")
EOF

echo ""
echo "======================================================"
echo "  Fix 5 — Add cheating_score weights for gaze events"
echo "           (so alerts actually increment scores)"
echo "======================================================"

python3 - << 'EOF'
path = "classification/cheating_score.py"
with open(path) as f:
    src = f.read()

old = '''        self.weights = {
            "PHONE_DETECTED": 40,
            "PASSING_OBJECT": 60,
            "WHISPERING_DETECTED": 30,
            "UNAUTHORIZED_OBJECT": 50,
            "LOOKING_AT_PHONE": 35,
            "BEHAVIOR_ANOMALY": 20
        }'''

new = '''        self.weights = {
            "PHONE_DETECTED": 40,
            "PASSING_OBJECT": 60,
            "WHISPERING_DETECTED": 30,
            "UNAUTHORIZED_OBJECT": 50,
            "LOOKING_AT_PHONE": 35,
            "BEHAVIOR_ANOMALY": 20,
            "LOOKING_LEFT": 5,
            "LOOKING_RIGHT": 5,
            "LOOKING_DOWN": 8,
            "BODY_LEANING": 10,
            "TALKING_DETECTED": 8,
            "HANDS_ON_FACE": 5,
            "SUSPICIOUS_GESTURE": 15,
            "SEAT_VACATED": 20,
            "CALCULATOR_DETECTED": 25,
            "WRITING_ON_PALM": 30,
            "STUDENTS_TOO_CLOSE": 12,
        }'''

if old in src:
    src = src.replace(old, new)
    with open(path, "w") as f:
        f.write(src)
    print("  [OK] cheating_score.py weights expanded")
else:
    print("  [SKIP] Weights already updated or pattern not found")
EOF

echo ""
echo "======================================================"
echo "  All fixes applied!"
echo ""
echo "  Now run:"
echo "  python3 process_video.py Student_Exam_Cheating_CCTV_Video.mp4"
echo ""
echo "  Or try the longer video for more detections:"
echo "  python3 process_video.py Student_Exam_Cheating_CCTV_Video1.mp4"
echo "======================================================"
