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
